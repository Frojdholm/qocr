from __future__ import annotations

import copy
import dataclasses
import importlib.resources
import pathlib

import cupy as cp
import cv2
import numpy as np
import onnxruntime as ort
import yaml

from qocr.rapid_word_results import (
    CalRecBoxes,
    Direction,
    WordInfo,
    WordResult,
    extract_word_info_from_ctc,
)


def _load_raw_kernel(resource_name: str, kernel_name: str) -> cp.RawKernel:
    kernel_code = (
        importlib.resources.files("qocr.kernels")
        .joinpath(resource_name)
        .read_text(encoding="utf-8")
    )
    return cp.RawKernel(kernel_code, kernel_name)


_WARP_KERNEL = _load_raw_kernel(
    "warp_perspective.cu",
    "batched_warp_perspective_tensor_kernel",
)


@dataclasses.dataclass(frozen=True)
class PreprocessConfig:
    crop_h: int = 48
    min_w: int = 160
    max_w: int = 3200
    batch_size: int = 16
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    sort_by_width: bool = True


@dataclasses.dataclass(frozen=True)
class PostprocessConfig:
    return_scores: bool = True
    return_word_box: bool = False
    return_single_char_box: bool = False


@dataclasses.dataclass(frozen=True)
class RecConfig:
    model_path: pathlib.Path | str
    dict_path: pathlib.Path | str | None = None
    device_id: int = 0
    preprocess: PreprocessConfig = PreprocessConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    use_tensorrt: bool = False
    trt_fp16: bool = True
    trt_cache_dir: pathlib.Path | None = None
    trt_max_batch_size: int = 16
    trt_max_width: int = 3200


@dataclasses.dataclass(frozen=True)
class RecResult:
    text: str
    score: float
    word_results: tuple[WordResult, ...] | None = None

    def __str__(self) -> str:
        return f"{self.text} ({self.score:.4f})"


def load_character_dict(dict_path: pathlib.Path) -> list[str]:
    """Loads character dictionary from YAML or text file.

    Index 0 is reserved for CTC blank token.
    """
    if dict_path.suffix.lower() in [".yml", ".yaml"]:
        with open(dict_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_chars = data.get("PostProcess", {}).get("character_dict")
        if raw_chars is None:
            raw_chars = data.get("character_dict")
        if raw_chars is None:
            raise KeyError(
                f"Could not find 'PostProcess.character_dict' in {dict_path}"
            )

        chars = [str(c) if c is not None else "" for c in raw_chars]
        return ["blank"] + chars + [" "]

    with open(dict_path, "r", encoding="utf-8") as f:
        chars = [line.strip("\r\n") for line in f]
    return ["blank"] + chars + [" "]


class RecSession:
    def __init__(
        self,
        config: RecConfig,
        stream: cp.cuda.Stream | None = None,
    ) -> None:
        self.config = config
        self.model_path = pathlib.Path(self.config.model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Recognition ONNX model file not found: {self.model_path}"
            )

        if self.config.dict_path is None:
            raise ValueError(
                "dict_path must be specified in RecConfig. Specify character dictionary file directly."
            )
        self.dict_path = pathlib.Path(self.config.dict_path)
        if not self.dict_path.is_file():
            raise FileNotFoundError(f"Dictionary file not found: {self.dict_path}")

        self.char_dict = load_character_dict(self.dict_path)

        with cp.cuda.Device(self.config.device_id):
            self.stream = (
                stream if stream is not None else cp.cuda.Stream(non_blocking=True)
            )

        cuda_provider = (
            "CUDAExecutionProvider",
            {
                "device_id": self.config.device_id,
                "user_compute_stream": str(self.stream.ptr),
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "cudnn_conv_use_max_workspace": "1",
                "cudnn_conv1d_pad_to_nc1d": "1",
                "do_copy_in_default_stream": "false",
            },
        )
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if self.config.use_tensorrt:
            trt_cache_dir = self.config.trt_cache_dir or (
                self.model_path.parent / ".trt_cache"
            )
            trt_cache_dir = pathlib.Path(trt_cache_dir)
            trt_cache_dir.mkdir(parents=True, exist_ok=True)

            temp_sess = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            in_name = temp_sess.get_inputs()[0].name
            del temp_sess

            crop_h = self.config.preprocess.crop_h
            min_w = self.config.preprocess.min_w
            max_w = max(self.config.preprocess.max_w, self.config.trt_max_width)
            max_bs = max(
                self.config.preprocess.batch_size, self.config.trt_max_batch_size
            )
            opt_bs = min(max_bs, 8)

            trt_options = {
                "device_id": self.config.device_id,
                "trt_fp16_enable": self.config.trt_fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_cache_dir),
                "trt_timing_cache_enable": True,
                "trt_profile_min_shapes": f"{in_name}:1x3x{crop_h}x{min_w}",
                "trt_profile_opt_shapes": f"{in_name}:{opt_bs}x3x{crop_h}x320",
                "trt_profile_max_shapes": f"{in_name}:{max_bs}x3x{crop_h}x{max_w}",
                "user_compute_stream": str(self.stream.ptr),
            }
            providers.append(("TensorrtExecutionProvider", trt_options))

        providers.append(cuda_provider)

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=providers,
        )

        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

        # Verify and align character dictionary length with model output dimension
        rec_out_shape = self.session.get_outputs()[0].shape
        if len(rec_out_shape) >= 3 and isinstance(rec_out_shape[-1], int):
            expected_dim = rec_out_shape[-1]
            if len(self.char_dict) != expected_dim:
                if len(self.char_dict) == expected_dim - 1:
                    self.char_dict.append(" ")
                elif (
                    len(self.char_dict) == expected_dim + 1
                    and self.char_dict[-1] == " "
                ):
                    self.char_dict.pop()

    @property
    def is_tensorrt(self) -> bool:
        return "TensorrtExecutionProvider" in self.session.get_providers()

    def run_batch(self, tensor: cp.ndarray) -> cp.ndarray:
        """Executes recognition ONNX graph directly on GPU memory with IO-Binding.

        Args:
            tensor: (B, 3, H, W) float32 GPU CuPy array in [-1.0, 1.0].

        Returns:
            gpu_rec_out: (B, seq_len, num_classes) float32 GPU CuPy array in VRAM.
        """
        with cp.cuda.Device(self.config.device_id), self.stream:
            io = self.session.io_binding()
            io.bind_input(
                self.in_name,
                "cuda",
                self.config.device_id,
                np.float32,
                tensor.shape,
                tensor.data.ptr,
            )
            io.bind_output(self.out_name, "cuda", self.config.device_id)
            self.session.run_with_iobinding(io)

            ort_out = io.get_outputs()[0]
            out_shape = ort_out.shape()
            mem = cp.cuda.UnownedMemory(
                ort_out.data_ptr(),
                int(np.prod(out_shape)) * 4,
                owner=(ort_out, io),
            )
            gpu_rec_out = cp.ndarray(
                shape=out_shape,
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(mem, 0),
            )

        return gpu_rec_out


def recognize_boxes(
    session: RecSession,
    image: cp.ndarray,
    boxes: np.ndarray,
    angles: cp.ndarray,
    return_word_box: bool | None = None,
    return_single_char_box: bool | None = None,
) -> list[RecResult]:
    """Extracts, normalizes, and recognizes text quadrilaterals directly in GPU VRAM.

    Intermediate text crops never touch CPU RAM or PCIe bus.

    Args:
        session: Active RecSession instance.
        image: (H, W, 3) uint8 GPU CuPy array.
        boxes: (N, 4, 2) float32 corner coordinates [TL, TR, BR, BL] on original image.
        angles: (N,) int32 orientation angles (0 or 180) CuPy array in GPU VRAM.
        return_word_box: Whether to extract word-level bounding boxes and confidences.
        return_single_char_box: Whether to split words into individual character boxes.

    Returns:
        list[RecResult]: Recognized text strings, confidence scores, and optional word results.
    """
    if len(boxes) == 0:
        return []

    should_return_word_box = (
        return_word_box
        if return_word_box is not None
        else session.config.postprocess.return_word_box
    )
    should_return_single_char_box = (
        return_single_char_box
        if return_single_char_box is not None
        else session.config.postprocess.return_single_char_box
    )

    with cp.cuda.Device(session.config.device_id), session.stream:
        img_h, img_w = image.shape[:2]

        crop_h = session.config.preprocess.crop_h
        min_w = session.config.preprocess.min_w
        max_w = session.config.preprocess.max_w
        batch_size = session.config.preprocess.batch_size
        mean_r, mean_g, mean_b = session.config.preprocess.mean
        std_r, std_g, std_b = session.config.preprocess.std

        w_top = np.linalg.norm(boxes[:, 1] - boxes[:, 0], axis=1)
        w_bot = np.linalg.norm(boxes[:, 2] - boxes[:, 3], axis=1)
        raw_w = np.maximum(w_top, w_bot)

        h_left = np.linalg.norm(boxes[:, 3] - boxes[:, 0], axis=1)
        h_right = np.linalg.norm(boxes[:, 2] - boxes[:, 1], axis=1)
        raw_h = np.maximum(np.maximum(h_left, h_right), 1.0)

        natural_widths = np.clip(
            np.ceil(crop_h * (raw_w / raw_h)).astype(int), 1, max_w
        )
        tensor_widths = np.maximum(natural_widths, min_w)

        if session.config.preprocess.sort_by_width and len(boxes) > 1:
            order = np.argsort(natural_widths)
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))

            boxes_proc = boxes[order]
            natural_widths = natural_widths[order]
            tensor_widths = tensor_widths[order]
            order_gpu = cp.asarray(order, dtype=cp.int32)
            angles_proc = angles[order_gpu]
        else:
            boxes_proc = boxes
            angles_proc = angles
            inv_order = None

        all_m_inv = np.empty((len(boxes_proc), 3, 3), dtype=np.float32)
        for idx, (b, nw) in enumerate(zip(boxes_proc, natural_widths)):
            dst_pts = np.array(
                [[0, 0], [nw, 0], [nw, crop_h], [0, crop_h]],
                dtype=np.float32,
            )
            all_m_inv[idx] = cv2.getPerspectiveTransform(dst_pts, b.astype(np.float32))

        all_m_inv_gpu = cp.asarray(all_m_inv, dtype=cp.float32)
        natural_widths_gpu = cp.asarray(natural_widths, dtype=cp.int32)

        batch_chunks = []

        for i in range(0, len(boxes_proc), batch_size):
            curr_bs = min(batch_size, len(boxes_proc) - i)
            chunk_tensor_widths = tensor_widths[i : i + curr_bs]
            max_batch_w = int(chunk_tensor_widths.max())

            m_inv_gpu = all_m_inv_gpu[i : i + curr_bs]
            widths_gpu = natural_widths_gpu[i : i + curr_bs]
            angles_gpu = angles_proc[i : i + curr_bs]

            batch_tensor = cp.zeros((curr_bs, 3, crop_h, max_batch_w), dtype=cp.float32)

            total_threads = curr_bs * crop_h * max_batch_w
            block_dim = 256
            grid_dim = (total_threads + block_dim - 1) // block_dim

            _WARP_KERNEL(
                (grid_dim,),
                (block_dim,),
                (
                    image,
                    m_inv_gpu,
                    widths_gpu,
                    angles_gpu,
                    batch_tensor,
                    np.int32(curr_bs),
                    np.int32(img_h),
                    np.int32(img_w),
                    np.int32(crop_h),
                    np.int32(max_batch_w),
                    np.float32(mean_r),
                    np.float32(mean_g),
                    np.float32(mean_b),
                    np.float32(std_r),
                    np.float32(std_g),
                    np.float32(std_b),
                ),
            )

            gpu_rec_out = session.run_batch(batch_tensor)

            gpu_token_ids = cp.argmax(gpu_rec_out, axis=-1)
            gpu_max_probs = cp.max(gpu_rec_out, axis=-1)

            batch_chunks.append(
                (
                    gpu_token_ids,
                    gpu_max_probs,
                    curr_bs,
                    max_batch_w,
                    natural_widths[i : i + curr_bs],
                )
            )

        all_results: list[RecResult] = []
        all_word_infos: list[WordInfo] = []

        for (
            gpu_token_ids,
            gpu_max_probs,
            curr_bs,
            max_batch_w,
            chunk_nws,
        ) in batch_chunks:
            cpu_tokens = cp.asnumpy(gpu_token_ids)
            cpu_probs = cp.asnumpy(gpu_max_probs)

            for b_idx in range(curr_bs):
                tokens = cpu_tokens[b_idx]
                probs = cpu_probs[b_idx]
                nw = int(chunk_nws[b_idx])

                if should_return_word_box:
                    text, score, word_info = extract_word_info_from_ctc(
                        tokens=tokens,
                        probs=probs,
                        char_dict=session.char_dict,
                        wh_ratio=float(nw),
                        max_wh_ratio=float(max_batch_w),
                        blank_idx=0,
                    )
                    all_results.append(RecResult(text=text, score=score))
                    all_word_infos.append(word_info)
                else:
                    chars: list[str] = []
                    char_probs: list[float] = []

                    for t_idx, token_id in enumerate(tokens):
                        if (
                            token_id != 0
                            and (t_idx == 0 or token_id != tokens[t_idx - 1])
                            and token_id < len(session.char_dict)
                        ):
                            chars.append(session.char_dict[token_id])
                            char_probs.append(float(probs[t_idx]))

                    text = "".join(chars)
                    score = float(np.mean(char_probs)) if char_probs else 0.0
                    all_results.append(RecResult(text=text, score=score))

        if should_return_word_box:
            cal_rec_boxes_op = CalRecBoxes()
            angles_proc_cpu = cp.asnumpy(angles_proc)
            updated_results: list[RecResult] = []
            for idx, (res, word_info, box, ang) in enumerate(
                zip(all_results, all_word_infos, boxes_proc, angles_proc_cpu)
            ):
                if not res.text or word_info.line_txt_len == 0:
                    updated_results.append(dataclasses.replace(res, word_results=()))
                    continue

                box_for_cal = np.roll(box, 2, axis=0) if ang == 180 else box

                img_crop_width = int(
                    max(
                        float(np.linalg.norm(box_for_cal[0] - box_for_cal[1])),
                        float(np.linalg.norm(box_for_cal[2] - box_for_cal[3])),
                    )
                )
                img_crop_height = int(
                    max(
                        float(np.linalg.norm(box_for_cal[0] - box_for_cal[3])),
                        float(np.linalg.norm(box_for_cal[1] - box_for_cal[2])),
                    )
                )
                direction = cal_rec_boxes_op.get_box_direction(box_for_cal)
                if direction == Direction.VERTICAL:
                    w, h = max(img_crop_height, 1), max(img_crop_width, 1)
                else:
                    w, h = max(img_crop_width, 1), max(img_crop_height, 1)

                img_box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                word_contents, word_boxes, word_confs = (
                    cal_rec_boxes_op.cal_ocr_word_box(
                        res.text,
                        img_box,
                        word_info,
                        should_return_single_char_box,
                    )
                )
                word_boxes = cal_rec_boxes_op.adjust_box_overlap(
                    copy.deepcopy(word_boxes)
                )
                word_boxes = cal_rec_boxes_op.reverse_rotate_crop_image(
                    copy.deepcopy(box_for_cal), word_boxes, direction
                )

                line_word_results = tuple(
                    WordResult(
                        text=w_txt,
                        score=round(float(w_score), 4),
                        box=np.asarray(w_box, dtype=np.int32),
                    )
                    for w_txt, w_score, w_box in zip(
                        word_contents, word_confs, word_boxes
                    )
                    if w_box is not None
                )
                updated_results.append(
                    dataclasses.replace(res, word_results=line_word_results)
                )
            all_results = updated_results

        if inv_order is not None:
            all_results = [all_results[idx] for idx in inv_order]

    return all_results
