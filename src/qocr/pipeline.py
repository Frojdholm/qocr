from __future__ import annotations

import dataclasses
import pathlib

import cupy as cp
import cv2
import numpy as np

from qocr import cls, det, rec
from qocr.rapid_word_results import WordResult


@dataclasses.dataclass(frozen=True)
class OCRResult:
    box: np.ndarray
    text: str
    score: float
    angle: int = 0
    angle_score: float = 1.0
    word_results: tuple[WordResult, ...] | None = None

    def __str__(self) -> str:
        base = f"{self.text:<30} | {self.angle}° ({self.angle_score:.2f}) | {self.score:.4f}"
        if self.word_results:
            words_str = ", ".join(f"'{w.text}'" for w in self.word_results)
            return f"{base} | words: [{words_str}]"
        return base


class OCRPipeline:
    """Zero-Copy GPU OCR Pipeline.

    Coordinates Text Detection (DBNet), Textline Orientation Classification (Cls),
    and Text Recognition (Rec/SVTR) in GPU VRAM on a shared CUDA stream with
    zero host-to-device and device-to-host bouncing of intermediate crops and feature maps.
    """

    def __init__(
        self,
        det_model_path: pathlib.Path | str,
        rec_model_path: pathlib.Path | str,
        cls_model_path: pathlib.Path | str | None = None,
        dict_path: pathlib.Path | str | None = None,
        device_id: int = 0,
        stream: cp.cuda.Stream | None = None,
        det_config: det.DBNetConfig | None = None,
        cls_config: cls.ClsConfig | None = None,
        rec_config: rec.RecConfig | None = None,
        use_tensorrt: bool = False,
        min_text_score: float = 0.5,
        drop_empty: bool = True,
        return_word_box: bool = False,
        return_single_char_box: bool = False,
    ) -> None:
        self.device_id = device_id
        self.min_text_score = min_text_score
        self.drop_empty = drop_empty
        self.return_word_box = return_word_box
        self.return_single_char_box = return_single_char_box
        with cp.cuda.Device(self.device_id):
            self.stream = (
                stream if stream is not None else cp.cuda.Stream(non_blocking=True)
            )

        if det_config is None:
            det_config = det.DBNetConfig(
                model_path=pathlib.Path(det_model_path),
                device_id=self.device_id,
                use_tensorrt=use_tensorrt,
            )
        elif use_tensorrt and not det_config.use_tensorrt:
            det_config = dataclasses.replace(det_config, use_tensorrt=True)
        self.det_session = det.DBNetSession(det_config, stream=self.stream)

        if cls_config is None and cls_model_path is not None:
            cls_config = cls.ClsConfig(
                model_path=pathlib.Path(cls_model_path),
                device_id=self.device_id,
                use_tensorrt=use_tensorrt,
            )
        elif cls_config is not None and use_tensorrt and not cls_config.use_tensorrt:
            cls_config = dataclasses.replace(cls_config, use_tensorrt=True)
        self.cls_session = (
            cls.ClsSession(cls_config, stream=self.stream)
            if cls_config is not None
            else None
        )

        if rec_config is None:
            rec_config = rec.RecConfig(
                model_path=pathlib.Path(rec_model_path),
                dict_path=pathlib.Path(dict_path) if dict_path else None,
                device_id=self.device_id,
                use_tensorrt=use_tensorrt,
                postprocess=rec.PostprocessConfig(
                    return_word_box=return_word_box,
                    return_single_char_box=return_single_char_box,
                ),
            )
        elif use_tensorrt and not rec_config.use_tensorrt:
            rec_config = dataclasses.replace(rec_config, use_tensorrt=True)
        self.rec_session = rec.RecSession(rec_config, stream=self.stream)

    def _prepare_gpu_image(
        self,
        image: str | pathlib.Path | np.ndarray | cp.ndarray,
    ) -> cp.ndarray:
        """Ensures the image is uploaded to GPU VRAM once as an RGB uint8 array."""
        if isinstance(image, (str, pathlib.Path)):
            im_bgr = cv2.imread(str(image))
            if im_bgr is None:
                raise FileNotFoundError(f"Failed to read image at: {image}")
            image = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        with cp.cuda.Device(self.device_id), self.stream:
            return cp.ascontiguousarray(cp.asarray(image, dtype=cp.uint8))

    def predict(
        self,
        image: str | pathlib.Path | np.ndarray | cp.ndarray,
        min_text_score: float | None = None,
        drop_empty: bool | None = None,
        return_word_box: bool | None = None,
        return_single_char_box: bool | None = None,
    ) -> tuple[list[OCRResult], cp.ndarray]:
        """Runs end-to-end zero-copy OCR on an input image.

        All intermediate text crop extractions, color conversions, normalizations,
        and 180° orientation corrections execute directly in GPU VRAM.

        Args:
            image: GPU CuPy array, CPU NumPy array, or file path.
            min_text_score: Minimum recognition score threshold (defaults to self.min_text_score).
            drop_empty: Whether to discard recognitions with empty or whitespace-only text (defaults to self.drop_empty).
            return_word_box: Whether to extract word-level bounding boxes and confidences (defaults to self.return_word_box).
            return_single_char_box: Whether to split words into individual character boxes (defaults to self.return_single_char_box).

        Returns:
            list[OCRResult] or tuple[list[OCRResult], cp.ndarray]:
                Detected text polygons, recognized strings, and confidences.
        """
        im_gpu = self._prepare_gpu_image(image)

        should_return_word_box = (
            return_word_box if return_word_box is not None else self.return_word_box
        )
        should_return_single_char_box = (
            return_single_char_box
            if return_single_char_box is not None
            else self.return_single_char_box
        )

        with cp.cuda.Device(self.device_id), self.stream:
            boxes, prob_map = det.detect(self.det_session, im_gpu)
            if len(boxes) == 0:
                return ([], prob_map)

            if self.cls_session is not None:
                cls_results, angles = cls.classify(self.cls_session, im_gpu, boxes)
            else:
                cls_results = [
                    cls.ClsResult(angle=0, score=1.0) for _ in range(len(boxes))
                ]
                angles = cp.zeros((len(boxes),), dtype=cp.int32)

            rec_results = rec.recognize_boxes(
                self.rec_session,
                im_gpu,
                boxes,
                angles,
                return_word_box=should_return_word_box,
                return_single_char_box=should_return_single_char_box,
            )

        score_thresh = (
            min_text_score if min_text_score is not None else self.min_text_score
        )
        should_drop_empty = drop_empty if drop_empty is not None else self.drop_empty

        results: list[OCRResult] = []
        for box, cr, rr in zip(boxes, cls_results, rec_results):
            if should_drop_empty and not rr.text.strip():
                continue
            if score_thresh > 0.0 and rr.score < score_thresh:
                continue

            results.append(
                OCRResult(
                    box=box,
                    text=rr.text,
                    score=rr.score,
                    angle=cr.angle,
                    angle_score=cr.score,
                    word_results=rr.word_results,
                )
            )

        return results, prob_map
