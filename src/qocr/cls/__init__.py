from __future__ import annotations

import dataclasses
import importlib.resources
import pathlib

import cupy as cp
import cv2
import numpy as np
import onnxruntime as ort
import yaml


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
    crop_h: int = 80
    crop_w: int = 160
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 64


@dataclasses.dataclass(frozen=True)
class PostprocessConfig:
    thresh: float = 0.9
    label_list: tuple[str, ...] | None = None


@dataclasses.dataclass(frozen=True)
class ClsConfig:
    model_path: pathlib.Path | str
    yaml_path: pathlib.Path | str | None = None
    device_id: int = 0
    preprocess: PreprocessConfig = PreprocessConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    use_tensorrt: bool = False
    trt_fp16: bool = True
    trt_cache_dir: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True)
class ClsResult:
    angle: int  # 0 or 180
    score: float

    def __str__(self) -> str:
        return f"{self.angle}° ({self.score:.4f})"


def load_cls_label_list(yml_path: pathlib.Path | None) -> list[str]:
    """Loads classification label list from YAML configuration if present."""
    if yml_path is not None and yml_path.is_file():
        with open(yml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        label_list = config.get("PostProcess", {}).get("Topk", {}).get("label_list")
        if label_list:
            return [str(l) for l in label_list]

    return ["0_degree", "180_degree"]


class ClsSession:
    def __init__(
        self,
        config: ClsConfig,
        stream: cp.cuda.Stream | None = None,
    ) -> None:
        self.config = config
        self.model_path = pathlib.Path(self.config.model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Classification ONNX model file not found: {self.model_path}"
            )
        self.yml_path = (
            pathlib.Path(self.config.yaml_path)
            if self.config.yaml_path is not None
            else None
        )

        if self.config.postprocess.label_list is not None:
            self.label_list = list(self.config.postprocess.label_list)
        elif self.yml_path is not None:
            self.label_list = load_cls_label_list(self.yml_path)
        else:
            self.label_list = ["0_degree", "180_degree"]

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
            crop_w = self.config.preprocess.crop_w
            max_bs = max(self.config.preprocess.batch_size, 64)
            opt_bs = min(max_bs, 16)

            trt_options = {
                "device_id": self.config.device_id,
                "trt_fp16_enable": self.config.trt_fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_cache_dir),
                "trt_timing_cache_enable": True,
                "trt_profile_min_shapes": f"{in_name}:1x3x{crop_h}x{crop_w}",
                "trt_profile_opt_shapes": f"{in_name}:{opt_bs}x3x{crop_h}x{crop_w}",
                "trt_profile_max_shapes": f"{in_name}:{max_bs}x3x{crop_h}x{crop_w}",
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

    @property
    def is_tensorrt(self) -> bool:
        return "TensorrtExecutionProvider" in self.session.get_providers()

    def run_batch(self, tensor: cp.ndarray) -> cp.ndarray:
        """Executes classification ONNX graph with IO-Binding directly on GPU memory.

        Args:
            tensor: (B, 3, 80, 160) float32 GPU CuPy array in VRAM.

        Returns:
            gpu_cls_out: (B, 2) float32 GPU CuPy array in VRAM.
        """
        curr_bs = tensor.shape[0]
        with cp.cuda.Device(self.config.device_id), self.stream:
            gpu_cls_out = cp.empty((curr_bs, 2), dtype=cp.float32)

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
                (curr_bs, 2),
                gpu_cls_out.data.ptr,
            )
            self.session.run_with_iobinding(io)

        return gpu_cls_out


def classify(
    session: ClsSession,
    image: cp.ndarray,
    boxes: np.ndarray,
) -> tuple[list[ClsResult], cp.ndarray]:
    """Warp quadrilateral text boxes to 80x160 and classify orientation (0° vs 180°).

    All perspective warping and normalization happen in GPU VRAM with zero intermediate host copies.

    Args:
        session: Active ClsSession instance.
        image: (H, W, 3) uint8 GPU CuPy array.
        boxes: (N, 4, 2) float32 corner coordinates [TL, TR, BR, BL] on original image.

    Returns:
        tuple[list[ClsResult], cp.ndarray]:
            Orientation results and (N,) int32 GPU angles tensor.
    """
    if len(boxes) == 0:
        return [], cp.empty((0,), dtype=cp.int32)

    with cp.cuda.Device(session.config.device_id), session.stream:
        img_h, img_w = image.shape[:2]

        crop_h = session.config.preprocess.crop_h
        crop_w = session.config.preprocess.crop_w
        batch_size = session.config.preprocess.batch_size
        mean_r, mean_g, mean_b = session.config.preprocess.mean
        std_r, std_g, std_b = session.config.preprocess.std
        cls_thresh = session.config.postprocess.thresh

        dst_pts = np.array(
            [[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]],
            dtype=np.float32,
        )

        all_m_inv = np.empty((len(boxes), 3, 3), dtype=np.float32)
        for idx, box in enumerate(boxes):
            all_m_inv[idx] = cv2.getPerspectiveTransform(
                dst_pts, box.astype(np.float32)
            )

        all_m_inv_gpu = cp.asarray(all_m_inv, dtype=cp.float32)
        all_widths_gpu = cp.full((len(boxes),), crop_w, dtype=cp.int32)
        all_angles_gpu = cp.zeros((len(boxes),), dtype=cp.int32)

        idx_180 = next((k for k, l in enumerate(session.label_list) if "180" in l), 1)

        batch_chunks = []

        for i in range(0, len(boxes), batch_size):
            curr_bs = min(batch_size, len(boxes) - i)

            m_inv_gpu = all_m_inv_gpu[i : i + curr_bs]
            widths_gpu = all_widths_gpu[i : i + curr_bs]
            angles_gpu = all_angles_gpu[i : i + curr_bs]
            batch_tensor = cp.empty((curr_bs, 3, crop_h, crop_w), dtype=cp.float32)

            total_pixels = curr_bs * crop_h * crop_w
            block_dim = 256
            grid_dim = (total_pixels + block_dim - 1) // block_dim

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
                    np.int32(crop_w),
                    np.float32(mean_r),
                    np.float32(mean_g),
                    np.float32(mean_b),
                    np.float32(std_r),
                    np.float32(std_g),
                    np.float32(std_b),
                ),
            )

            gpu_cls_out = session.run_batch(batch_tensor)

            exp_out = cp.exp(gpu_cls_out - cp.max(gpu_cls_out, axis=-1, keepdims=True))
            probs = exp_out / cp.sum(exp_out, axis=-1, keepdims=True)
            pred_labels = cp.argmax(probs, axis=-1)
            pred_scores = cp.max(probs, axis=-1)

            is_180 = (pred_labels == idx_180) & (pred_scores >= cls_thresh)
            chunk_gpu_angles = cp.where(is_180, cp.int32(180), cp.int32(0))

            batch_chunks.append((pred_labels, pred_scores, chunk_gpu_angles, curr_bs))

        all_pred_scores = cp.concatenate([b[1] for b in batch_chunks], axis=0)
        gpu_angles = cp.concatenate([b[2] for b in batch_chunks], axis=0)

        cpu_scores = cp.asnumpy(all_pred_scores)
        cpu_angles = cp.asnumpy(gpu_angles)

        all_results = [
            ClsResult(angle=int(cpu_angles[idx]), score=float(cpu_scores[idx]))
            for idx in range(len(boxes))
        ]

    return all_results, gpu_angles


classify_boxes = classify
