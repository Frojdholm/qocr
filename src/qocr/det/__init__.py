from __future__ import annotations

import dataclasses
import pathlib

import cupy as cp
import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from cupyx.scipy import ndimage
from shapely import geometry


@dataclasses.dataclass(frozen=True)
class PreprocessConfig:
    max_side_len: int = 736
    limit_type: str = "max"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclasses.dataclass(frozen=True)
class PostprocessConfig:
    thresh: float = 0.3
    box_thresh: float = 0.5
    unclip_ratio: float = 1.6
    min_size: int = 3
    calc_box_score: bool = True
    use_dilation: bool = True


@dataclasses.dataclass(frozen=True)
class DBNetConfig:
    model_path: pathlib.Path
    device_id: int = 0
    preprocess: PreprocessConfig = PreprocessConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    use_tensorrt: bool = False
    trt_fp16: bool = True
    trt_cache_dir: pathlib.Path | None = None
    trt_max_side_len: int = 1536


@dataclasses.dataclass(frozen=True)
class DBNetInput:
    tensor: cp.ndarray  # (1, 3, resize_h, resize_w) float32 GPU array
    orig_h: int
    orig_w: int
    resize_h: int
    resize_w: int

    @property
    def ratio_h(self) -> float:
        return self.resize_h / float(self.orig_h)

    @property
    def ratio_w(self) -> float:
        return self.resize_w / float(self.orig_w)


def preprocess_image(
    image: cp.ndarray,
    det_mean: cp.ndarray,
    det_std: cp.ndarray,
    max_side_len: int,
    limit_type: str = "max",
    max_limit: int | None = None,
) -> DBNetInput:
    """Rescales RGB image to 32px multiples and normalizes into an NCHW CuPy tensor."""
    orig_h, orig_w = image.shape[:2]

    if limit_type == "max":
        ratio = (
            float(max_side_len) / max(orig_h, orig_w)
            if max(orig_h, orig_w) > max_side_len
            else 1.0
        )
    else:
        ratio = (
            float(max_side_len) / min(orig_h, orig_w)
            if min(orig_h, orig_w) < max_side_len
            else 1.0
        )

    if max_limit is not None and max(orig_h, orig_w) * ratio > max_limit:
        ratio = float(max_limit) / max(orig_h, orig_w)

    resize_h = max(int(round((orig_h * ratio) / 32.0) * 32), 32)
    resize_w = max(int(round((orig_w * ratio) / 32.0) * 32), 32)

    if max_limit is not None:
        max_limit_aligned = max(32, (max_limit // 32) * 32)
        resize_h = min(resize_h, max_limit_aligned)
        resize_w = min(resize_w, max_limit_aligned)

    zoom = (resize_h / orig_h, resize_w / orig_w, 1)
    resized = ndimage.zoom(image, zoom, order=1)

    chw_im = resized.transpose(2, 0, 1)[cp.newaxis, ...]
    det_input = (chw_im.astype(cp.float32) / 255.0 - det_mean) / det_std
    det_input = cp.ascontiguousarray(det_input, dtype=cp.float32)

    return DBNetInput(
        tensor=det_input,
        orig_h=orig_h,
        orig_w=orig_w,
        resize_h=resize_h,
        resize_w=resize_w,
    )


class DBNetSession:
    def __init__(
        self,
        config: DBNetConfig,
        stream: cp.cuda.Stream | None = None,
    ) -> None:
        self.config = config
        self.model_path = pathlib.Path(self.config.model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Detection ONNX model file not found: {self.model_path}"
            )

        with cp.cuda.Device(self.config.device_id):
            self.det_mean = cp.array(
                self.config.preprocess.mean, dtype=cp.float32
            ).reshape((1, 3, 1, 1))
            self.det_std = cp.array(
                self.config.preprocess.std, dtype=cp.float32
            ).reshape((1, 3, 1, 1))
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

            opt_side = self.config.preprocess.max_side_len
            self.max_side = max(self.config.trt_max_side_len, opt_side)

            trt_options = {
                "device_id": self.config.device_id,
                "trt_fp16_enable": self.config.trt_fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_cache_dir),
                "trt_timing_cache_enable": True,
                "trt_profile_min_shapes": f"{in_name}:1x3x32x32",
                "trt_profile_opt_shapes": f"{in_name}:1x3x{opt_side}x{opt_side}",
                "trt_profile_max_shapes": f"{in_name}:1x3x{self.max_side}x{self.max_side}",
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
        self.max_side = max(
            self.config.trt_max_side_len, self.config.preprocess.max_side_len
        )

    @property
    def is_tensorrt(self) -> bool:
        return "TensorrtExecutionProvider" in self.session.get_providers()

    def run(self, tensor: cp.ndarray) -> cp.ndarray:
        """Executes graph with IO-Binding.

        Expects float32 NCHW tensor on GPU; writes directly to CuPy GPU memory.
        """
        out_shape = (tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])

        with cp.cuda.Device(self.config.device_id), self.stream:
            gpu_prob_map = cp.empty(out_shape, dtype=cp.float32)

            io = self.session.io_binding()
            io.bind_input(
                self.in_name,
                "cuda",
                self.config.device_id,
                np.float32,
                tensor.shape,
                tensor.data.ptr,
            )
            io.bind_output(
                self.out_name,
                "cuda",
                self.config.device_id,
                np.float32,
                out_shape,
                gpu_prob_map.data.ptr,
            )
            self.session.run_with_iobinding(io)

        return gpu_prob_map


def calculate_box_score(bitmap: np.ndarray, box: np.ndarray) -> float:
    h, w = bitmap.shape[:2]
    xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

    if xmin >= xmax or ymin >= ymax:
        return 0.0

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box_shifted = box - np.array([xmin, ymin])
    cv2.fillPoly(mask, [box_shifted.astype(np.int32)], 1)
    return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask=mask)[0])


def postprocess_dbnet(
    prob_map: np.ndarray,
    orig_h: int,
    orig_w: int,
    ratio_h: float,
    ratio_w: float,
    config: PostprocessConfig,
) -> np.ndarray:
    mask = (prob_map > config.thresh).astype(np.uint8)
    if config.use_dilation:
        kernel = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        bbox = cv2.minAreaRect(contour)
        if min(bbox[1]) < config.min_size:
            continue

        pts = cv2.boxPoints(bbox)
        if (
            config.calc_box_score
            and calculate_box_score(prob_map, pts) < config.box_thresh
        ):
            continue

        poly = geometry.Polygon(pts)
        if poly.length == 0:
            continue
        dist = poly.area * config.unclip_ratio / (poly.length + 1e-6)

        offset = pyclipper.PyclipperOffset()
        offset.AddPath(
            pts.astype(np.int32),
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON,
        )
        expanded = offset.Execute(dist)
        if not expanded:
            continue

        exp_pts = cv2.boxPoints(cv2.minAreaRect(np.array(expanded[0])))
        if min(cv2.minAreaRect(exp_pts)[1]) < config.min_size + 2:
            continue

        exp_pts[:, 0] = np.clip(np.round(exp_pts[:, 0] / ratio_w), 0, orig_w)
        exp_pts[:, 1] = np.clip(np.round(exp_pts[:, 1] / ratio_h), 0, orig_h)

        # Order corners clockwise: TL, TR, BR, BL
        s = exp_pts.sum(axis=1)
        diff = np.diff(exp_pts, axis=1)
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = exp_pts[np.argmin(s)]
        rect[2] = exp_pts[np.argmax(s)]
        rect[1] = exp_pts[np.argmin(diff)]
        rect[3] = exp_pts[np.argmax(diff)]
        boxes.append(rect)

    if not boxes:
        return np.empty((0, 4, 2), dtype=np.float32)

    return sort_boxes(np.array(boxes, dtype=np.float32))


def sort_boxes(boxes: np.ndarray, y_threshold: float = 10.0) -> np.ndarray:
    """Groups text boxes into lines within y_threshold and sorts left-to-right."""
    if len(boxes) == 0:
        return boxes

    # Step 1: Stable sort by y (top to bottom)
    y_coords = boxes[:, 0, 1]
    y_order = np.argsort(y_coords, kind="stable")
    boxes_y_sorted = boxes[y_order]
    y_sorted = y_coords[y_order]

    # Step 2: Assign line IDs based on adjacent y differences
    dy = np.diff(y_sorted)
    line_increments = (dy >= y_threshold).astype(np.int32)
    line_ids = np.concatenate([[0], np.cumsum(line_increments)])

    # Step 3: Within each line group, sort by x (left to right)
    x_coords = boxes_y_sorted[:, 0, 0]
    final_order_in_y_sorted = np.lexsort((x_coords, line_ids))

    return boxes_y_sorted[final_order_in_y_sorted]


def detect(
    session: DBNetSession,
    image: cp.ndarray,
) -> tuple[np.ndarray, cp.ndarray]:
    """Runs zero-copy DBNet text detection on a GPU image (HWC CuPy array).

    Returns:
        tuple[np.ndarray, cp.ndarray]:
            - boxes: (N, 4, 2) float32 corner coordinates on original image
            - prob_map: (H, W) float32 GPU CuPy array at model resolution
    """
    with cp.cuda.Device(session.config.device_id), session.stream:
        prep_data = preprocess_image(
            image,
            session.det_mean,
            session.det_std,
            session.config.preprocess.max_side_len,
            session.config.preprocess.limit_type,
            max_limit=session.max_side if session.is_tensorrt else None,
        )
        gpu_prob = session.run(prep_data.tensor)
        gpu_prob_2d = gpu_prob[0, 0]
        prob_map_cpu = cp.asnumpy(gpu_prob_2d)

    boxes = postprocess_dbnet(
        prob_map=prob_map_cpu,
        orig_h=prep_data.orig_h,
        orig_w=prep_data.orig_w,
        ratio_h=prep_data.ratio_h,
        ratio_w=prep_data.ratio_w,
        config=session.config.postprocess,
    )

    return boxes, gpu_prob_2d
