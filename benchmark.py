from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Any

import cupy as cp
import cv2
import numpy as np

from qocr import cls, det, rec
from qocr.pipeline import OCRPipeline

REPO_ROOT = pathlib.Path(__file__).resolve().parent


def _default_path(subpath: str) -> str:
    return str(REPO_ROOT / subpath)


def _init_rapidocr(
    det_model: pathlib.Path | str,
    rec_model: pathlib.Path | str,
    cls_model: pathlib.Path | str,
    dict_file: pathlib.Path | str,
) -> Any:
    """Initializes RapidOCR."""
    import rapidocr
    from rapidocr.utils import typings

    return rapidocr.RapidOCR(
        params={
            "Global.use_det": True,
            "Global.use_rec": True,
            "Global.use_cls": True,
            "Det.model_path": str(det_model),
            "Rec.model_path": str(rec_model),
            "Cls.model_path": str(cls_model),
            "Rec.rec_keys_path": str(dict_file),
            "Det.engine_type": typings.EngineType.ONNXRUNTIME,
            "Rec.engine_type": typings.EngineType.ONNXRUNTIME,
            "Cls.engine_type": typings.EngineType.ONNXRUNTIME,
            "EngineConfig.onnxruntime.use_cuda": True,
        }
    )


def _parse_rapidocr_results(out: Any) -> list[dict[str, Any]]:
    """Normalizes RapidOCR output into a list of {text, score, box} dicts."""
    if not out:
        return []
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "txts") and out.txts is not None:
        boxes = getattr(out, "boxes", None) or []
        scores = getattr(out, "scores", None) or []
        return [
            {
                "text": str(text),
                "score": round(float(scores[i]), 4) if i < len(scores) else 1.0,
                "box": boxes[i] if i < len(boxes) else None,
            }
            for i, text in enumerate(out.txts)
        ]
    if isinstance(out, list):
        return [
            {
                "text": str(item[1]),
                "score": round(float(item[2]), 4) if len(item) > 2 else 1.0,
                "box": item[0],
            }
            for item in out
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
    return []


def get_gpu_info(device_id: int = 0) -> dict[str, Any]:
    """Retrieves metadata and current memory stats for the specified CUDA device."""
    with cp.cuda.Device(device_id):
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props.get("name", b"Unknown GPU")
        name = (
            raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        )
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        compute_capability = f"{props.get('major', 0)}.{props.get('minor', 0)}"

    return {
        "device_id": device_id,
        "name": name,
        "compute_capability": compute_capability,
        "free_vram_mb": free_mem / (1024 * 1024),
        "total_vram_mb": total_mem / (1024 * 1024),
    }


@dataclasses.dataclass(frozen=True)
class BenchmarkStats:
    name: str
    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    fps: float

    @classmethod
    def from_timings(cls, name: str, timings_ms: list[float]) -> BenchmarkStats:
        if not timings_ms:
            return cls(
                name=name,
                count=0,
                mean_ms=0.0,
                std_ms=0.0,
                min_ms=0.0,
                median_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
                fps=0.0,
            )

        arr = np.array(timings_ms, dtype=np.float64)
        mean_val = float(np.mean(arr))
        return cls(
            name=name,
            count=len(arr),
            mean_ms=mean_val,
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            median_ms=float(np.median(arr)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(np.max(arr)),
            fps=1000.0 / mean_val if mean_val > 0.0 else 0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "fps": round(self.fps, 2),
        }


def run_benchmark(
    image_path: pathlib.Path | str,
    det_model_path: pathlib.Path | str,
    rec_model_path: pathlib.Path | str,
    cls_model_path: pathlib.Path | str,
    dict_path: pathlib.Path | str,
    device_id: int = 0,
    iterations: int = 50,
    warmup: int = 10,
    profile_stages: bool = True,
    include_disk: bool = True,
    compare_rapidocr: bool = True,
    use_tensorrt: bool = False,
    sort_by_width: bool = True,
    min_text_score: float = 0.5,
    drop_empty: bool = True,
    det_thresh: float | None = None,
    det_box_thresh: float | None = None,
    det_unclip_ratio: float | None = None,
    rapid_det_model_path: pathlib.Path | str | None = None,
    rapid_rec_model_path: pathlib.Path | str | None = None,
    rapid_cls_model_path: pathlib.Path | str | None = None,
    rapid_dict_path: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    image_path = pathlib.Path(image_path)
    im_bgr = cv2.imread(str(image_path))
    img_h, img_w = im_bgr.shape[:2]
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    rapid_det = rapid_det_model_path or det_model_path
    rapid_rec = rapid_rec_model_path or rec_model_path
    rapid_cls = rapid_cls_model_path or cls_model_path
    rapid_dict = rapid_dict_path or dict_path

    rec_cfg = rec.RecConfig(
        model_path=rec_model_path,
        dict_path=dict_path,
        device_id=device_id,
        use_tensorrt=use_tensorrt,
        preprocess=rec.PreprocessConfig(sort_by_width=sort_by_width),
    )

    det_post_kwargs: dict[str, Any] = {}
    if det_thresh is not None:
        det_post_kwargs["thresh"] = det_thresh
    if det_box_thresh is not None:
        det_post_kwargs["box_thresh"] = det_box_thresh
    if det_unclip_ratio is not None:
        det_post_kwargs["unclip_ratio"] = det_unclip_ratio

    det_cfg = None
    if det_post_kwargs:
        det_cfg = det.DBNetConfig(
            model_path=det_model_path,
            device_id=device_id,
            use_tensorrt=use_tensorrt,
            postprocess=det.PostprocessConfig(**det_post_kwargs),
        )

    pipeline = OCRPipeline(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
        dict_path=dict_path,
        device_id=device_id,
        det_config=det_cfg,
        rec_config=rec_cfg,
        use_tensorrt=use_tensorrt,
        min_text_score=min_text_score,
        drop_empty=drop_empty,
    )

    gpu_info = get_gpu_info(device_id)
    im_gpu = pipeline._prepare_gpu_image(im_rgb)

    # Warmup
    with cp.cuda.Device(device_id):
        pipeline.stream.synchronize()
    for _ in range(warmup):
        pipeline.predict(im_gpu)
    with cp.cuda.Device(device_id):
        pipeline.stream.synchronize()

    initial_results, _ = pipeline.predict(im_gpu)
    with cp.cuda.Device(device_id):
        pipeline.stream.synchronize()

    stats_list: list[BenchmarkStats] = []
    timings_record: dict[str, list[float]] = {}

    # Stage Breakdown Profiling
    if profile_stages:
        upload_timings: list[float] = []
        det_timings: list[float] = []
        cls_timings: list[float] = []
        rec_timings: list[float] = []

        for _ in range(iterations):
            with cp.cuda.Device(device_id):
                pipeline.stream.synchronize()
            t0 = time.perf_counter()
            _ = pipeline._prepare_gpu_image(im_rgb)
            with cp.cuda.Device(device_id):
                pipeline.stream.synchronize()
            upload_timings.append((time.perf_counter() - t0) * 1000.0)

            with cp.cuda.Device(device_id):
                pipeline.stream.synchronize()
            t0 = time.perf_counter()
            boxes, _ = det.detect(pipeline.det_session, im_gpu)
            with cp.cuda.Device(device_id):
                pipeline.stream.synchronize()
            det_timings.append((time.perf_counter() - t0) * 1000.0)

            if len(boxes) > 0:
                with cp.cuda.Device(device_id):
                    pipeline.stream.synchronize()
                t0 = time.perf_counter()
                _, angles = cls.classify(pipeline.cls_session, im_gpu, boxes)
                with cp.cuda.Device(device_id):
                    pipeline.stream.synchronize()
                cls_timings.append((time.perf_counter() - t0) * 1000.0)

                with cp.cuda.Device(device_id):
                    pipeline.stream.synchronize()
                t0 = time.perf_counter()
                _ = rec.recognize_boxes(pipeline.rec_session, im_gpu, boxes, angles)
                with cp.cuda.Device(device_id):
                    pipeline.stream.synchronize()
                rec_timings.append((time.perf_counter() - t0) * 1000.0)
            else:
                cls_timings.append(0.0)
                rec_timings.append(0.0)

        timings_record["h2d_upload"] = upload_timings
        timings_record["detection"] = det_timings
        timings_record["classification"] = cls_timings
        timings_record["recognition"] = rec_timings
        stats_list.append(
            BenchmarkStats.from_timings("Host-to-Device Upload", upload_timings)
        )
        stats_list.append(
            BenchmarkStats.from_timings("Text Detection (DBNet)", det_timings)
        )
        stats_list.append(
            BenchmarkStats.from_timings("Orientation Cls (LCNet)", cls_timings)
        )
        stats_list.append(
            BenchmarkStats.from_timings("Text Recognition (SVTR)", rec_timings)
        )

    # End-to-End Pipeline (VRAM Resident)
    e2e_vram_timings: list[float] = []
    for _ in range(iterations):
        with cp.cuda.Device(device_id):
            pipeline.stream.synchronize()
        t0 = time.perf_counter()
        _ = pipeline.predict(im_gpu)
        with cp.cuda.Device(device_id):
            pipeline.stream.synchronize()
        e2e_vram_timings.append((time.perf_counter() - t0) * 1000.0)

    timings_record["e2e_vram"] = e2e_vram_timings
    e2e_vram_stats = BenchmarkStats.from_timings(
        "End-to-End (VRAM Resident)", e2e_vram_timings
    )
    stats_list.append(e2e_vram_stats)

    # End-to-End Pipeline from Disk
    e2e_disk_stats = None
    if include_disk:
        e2e_disk_timings: list[float] = []
        for _ in range(iterations):
            with cp.cuda.Device(device_id):
                pipeline.stream.synchronize()
            t0 = time.perf_counter()
            _ = pipeline.predict(image_path)
            with cp.cuda.Device(device_id):
                pipeline.stream.synchronize()
            e2e_disk_timings.append((time.perf_counter() - t0) * 1000.0)

        timings_record["e2e_disk"] = e2e_disk_timings
        e2e_disk_stats = BenchmarkStats.from_timings(
            "End-to-End (from Disk)", e2e_disk_timings
        )
        stats_list.append(e2e_disk_stats)

    pool = cp.get_default_memory_pool()
    used_vram_mb = pool.used_bytes() / (1024 * 1024)
    total_allocated_mb = pool.total_bytes() / (1024 * 1024)

    # RapidOCR Comparison Benchmark
    rapid_report: dict[str, Any] | None = None
    if compare_rapidocr:
        rapid_engine = _init_rapidocr(
            det_model=rapid_det,
            rec_model=rapid_rec,
            cls_model=rapid_cls,
            dict_file=rapid_dict,
        )

        with cp.cuda.Device(device_id):
            cp.cuda.Stream.null.synchronize()
        for _ in range(warmup):
            rapid_engine(im_bgr)
        with cp.cuda.Device(device_id):
            cp.cuda.Stream.null.synchronize()

        initial_rapid_results = _parse_rapidocr_results(rapid_engine(im_bgr))
        with cp.cuda.Device(device_id):
            cp.cuda.Stream.null.synchronize()

        rapid_mem_timings: list[float] = []
        for _ in range(iterations):
            with cp.cuda.Device(device_id):
                cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            _ = rapid_engine(im_bgr)
            with cp.cuda.Device(device_id):
                cp.cuda.Stream.null.synchronize()
            rapid_mem_timings.append((time.perf_counter() - t0) * 1000.0)

        rapid_mem_stats = BenchmarkStats.from_timings(
            "RapidOCR (In-Memory CPU)", rapid_mem_timings
        )
        rapid_stats_list = [rapid_mem_stats]

        rapid_disk_stats = None
        if include_disk:
            rapid_disk_timings: list[float] = []
            for _ in range(iterations):
                with cp.cuda.Device(device_id):
                    cp.cuda.Stream.null.synchronize()
                t0 = time.perf_counter()
                _ = rapid_engine(str(image_path))
                with cp.cuda.Device(device_id):
                    cp.cuda.Stream.null.synchronize()
                rapid_disk_timings.append((time.perf_counter() - t0) * 1000.0)

            rapid_disk_stats = BenchmarkStats.from_timings(
                "RapidOCR (from Disk)", rapid_disk_timings
            )
            rapid_stats_list.append(rapid_disk_stats)

        disk_speedup = (
            rapid_disk_stats.mean_ms / e2e_disk_stats.mean_ms
            if rapid_disk_stats and e2e_disk_stats and e2e_disk_stats.mean_ms > 0
            else None
        )
        in_memory_speedup = (
            rapid_mem_stats.mean_ms / e2e_vram_stats.mean_ms
            if rapid_mem_stats and e2e_vram_stats and e2e_vram_stats.mean_ms > 0
            else None
        )
        latency_reduction_pct = (
            (1.0 - (e2e_vram_stats.mean_ms / rapid_mem_stats.mean_ms)) * 100.0
            if rapid_mem_stats and rapid_mem_stats.mean_ms > 0
            else None
        )

        rapid_report = {
            "available": True,
            "models": {
                "det_model": str(rapid_det),
                "rec_model": str(rapid_rec),
                "cls_model": str(rapid_cls),
                "dict_file": str(rapid_dict),
            },
            "num_detections": len(initial_rapid_results),
            "sample_detections": [
                {"text": r["text"], "score": r["score"]}
                for r in initial_rapid_results[:5]
            ],
            "stats": [s.to_dict() for s in rapid_stats_list],
            "stats_objects": rapid_stats_list,
            "comparison": {
                "disk_speedup": round(disk_speedup, 2) if disk_speedup else None,
                "in_memory_speedup": round(in_memory_speedup, 2)
                if in_memory_speedup
                else None,
                "latency_reduction_pct": round(latency_reduction_pct, 1)
                if latency_reduction_pct is not None
                else None,
            },
        }

    return {
        "device": gpu_info,
        "input": {
            "path": str(image_path),
            "height": img_h,
            "width": img_w,
            "num_detections": len(initial_results),
            "sample_detections": [
                {"text": r.text, "score": round(r.score, 4), "angle": r.angle}
                for r in initial_results[:5]
            ],
        },
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "profile_stages": profile_stages,
            "include_disk": include_disk,
            "compare_rapidocr": compare_rapidocr,
            "det_model": str(det_model_path),
            "cls_model": str(cls_model_path),
            "rec_model": str(rec_model_path),
            "dict_path": str(dict_path),
            "use_tensorrt": use_tensorrt,
            "sort_by_width": sort_by_width,
        },
        "memory": {
            "cupy_used_mb": round(used_vram_mb, 2),
            "cupy_pool_total_mb": round(total_allocated_mb, 2),
        },
        "stats": [s.to_dict() for s in stats_list],
        "stats_objects": stats_list,
        "rapidocr": rapid_report,
    }


def print_benchmark_table(report: dict[str, Any]) -> None:
    """Formats and prints the benchmark report to stdout with a clean ASCII table."""
    dev = report["device"]
    inp = report["input"]
    cfg = report["config"]
    mem = report["memory"]
    stats_list: list[BenchmarkStats] = report["stats_objects"]
    rapid_report = report.get("rapidocr")

    sep_wide = "=" * 88
    sep_thin = "-" * 88

    print("\n" + sep_wide)
    print(f"{'qOCR Pipeline Benchmark':^88}")
    print(sep_wide)
    print(
        f"Device:       {dev['name']} (ID: {dev['device_id']}, Compute: {dev['compute_capability']})"
    )
    print(
        f"VRAM Pool:    {mem['cupy_used_mb']:.1f} MB used / {mem['cupy_pool_total_mb']:.1f} MB allocated"
    )
    print(f"Input Image:  {inp['path']} ({inp['width']}x{inp['height']})")
    print(f"Detections:   {inp['num_detections']} text boxes detected (qOCR)")
    print(f"Models:       Det:  {pathlib.Path(cfg['det_model']).name}")
    print(f"              Cls:  {pathlib.Path(cfg['cls_model']).name}")
    print(f"              Rec:  {pathlib.Path(cfg['rec_model']).name}")
    print(f"              Dict: {pathlib.Path(cfg['dict_path']).name}")
    print(f"Benchmark:    {cfg['iterations']} iterations (warmup: {cfg['warmup']})")
    print(sep_thin)

    headers = f"{'Stage (qOCR)':<26} {'Mean':>9} {'Std':>8} {'Min':>8} {'Median':>9} {'P95':>8} {'Max':>8} {'FPS':>8}"
    print(headers)
    print(
        f"{'':<26} {'(ms)':>9} {'(ms)':>8} {'(ms)':>8} {'(ms)':>9} {'(ms)':>8} {'(ms)':>8} {'':>8}"
    )
    print(sep_thin)

    for s in stats_list:
        if "End-to-End" in s.name and s != stats_list[0]:
            print(sep_thin)
        print(
            f"{s.name:<26} "
            f"{s.mean_ms:>9.2f} "
            f"{s.std_ms:>8.2f} "
            f"{s.min_ms:>8.2f} "
            f"{s.median_ms:>9.2f} "
            f"{s.p95_ms:>8.2f} "
            f"{s.max_ms:>8.2f} "
            f"{s.fps:>8.1f}"
        )
    print(sep_wide)

    # RapidOCR stats
    if (
        rapid_report
        and rapid_report.get("available")
        and "stats_objects" in rapid_report
    ):
        rapid_stats: list[BenchmarkStats] = rapid_report["stats_objects"]
        rapid_models = rapid_report["models"]
        print("\n" + sep_wide)
        print(f"{'RapidOCR Baseline Benchmark':^88}")
        print(sep_wide)
        print("Engine:       RapidOCR (ONNX Runtime with CUDAExecutionProvider)")
        print(f"Detections:   {rapid_report['num_detections']} text boxes detected")
        print(f"Models:       Det:  {pathlib.Path(rapid_models['det_model']).name}")
        print(f"              Cls:  {pathlib.Path(rapid_models['cls_model']).name}")
        print(f"              Rec:  {pathlib.Path(rapid_models['rec_model']).name}")
        print(f"              Dict: {pathlib.Path(rapid_models['dict_file']).name}")
        print(sep_thin)
        headers_rapid = f"{'Stage (RapidOCR)':<26} {'Mean':>9} {'Std':>8} {'Min':>8} {'Median':>9} {'P95':>8} {'Max':>8} {'FPS':>8}"
        print(headers_rapid)
        print(
            f"{'':<26} {'(ms)':>9} {'(ms)':>8} {'(ms)':>8} {'(ms)':>9} {'(ms)':>8} {'(ms)':>8} {'':>8}"
        )
        print(sep_thin)
        for s in rapid_stats:
            print(
                f"{s.name:<26} "
                f"{s.mean_ms:>9.2f} "
                f"{s.std_ms:>8.2f} "
                f"{s.min_ms:>8.2f} "
                f"{s.median_ms:>9.2f} "
                f"{s.p95_ms:>8.2f} "
                f"{s.max_ms:>8.2f} "
                f"{s.fps:>8.1f}"
            )
        print(sep_wide)

        # Head-to-Head Comparison
        comp = rapid_report.get("comparison", {})
        qocr_vram = next((s for s in stats_list if "VRAM Resident" in s.name), None)
        qocr_disk = next((s for s in stats_list if "from Disk" in s.name), None)
        rapid_mem = next((s for s in rapid_stats if "In-Memory" in s.name), None)
        rapid_disk = next((s for s in rapid_stats if "from Disk" in s.name), None)

        print("\n" + sep_wide)
        print(f"{'qOCR vs RapidOCR Comparison':^88}")
        print(sep_wide)
        print(
            f"{'Pipeline Mode':<26} {'qOCR':>12} {'RapidOCR':>14} {'Speedup / Difference':>32}"
        )
        print(sep_thin)

        if qocr_disk and rapid_disk and comp.get("disk_speedup"):
            pct = 100.0 * (1.0 - qocr_disk.mean_ms / rapid_disk.mean_ms)
            disk_desc = f"{comp['disk_speedup']:.2f}x faster ({pct:.1f}% less time)"
            print(
                f"{'End-to-End (from Disk)':<26} {qocr_disk.mean_ms:>9.2f} ms {rapid_disk.mean_ms:>11.2f} ms {disk_desc:>32}"
            )

        if qocr_vram and rapid_mem and comp.get("in_memory_speedup"):
            pct = 100.0 * (1.0 - qocr_vram.mean_ms / rapid_mem.mean_ms)
            mem_desc = f"{comp['in_memory_speedup']:.2f}x faster ({pct:.1f}% less time)"
            print(
                f"{'In-Memory Pipeline':<26} {qocr_vram.mean_ms:>9.2f} ms {rapid_mem.mean_ms:>11.2f} ms {mem_desc:>32}"
            )

            fps_diff = qocr_vram.fps - rapid_mem.fps
            fps_pct = 100.0 * (fps_diff / rapid_mem.fps) if rapid_mem.fps > 0 else 0.0
            fps_desc = f"+{fps_diff:.1f} FPS (+{fps_pct:.1f}%)"
            print(
                f"{'Throughput (In-Memory)':<26} {qocr_vram.fps:>9.1f} FPS {rapid_mem.fps:>10.1f} FPS {fps_desc:>32}"
            )

        print(sep_thin)
        print(
            f"Detected Text Count:   qOCR: {inp['num_detections']} boxes | RapidOCR: {rapid_report['num_detections']} boxes"
        )
        print(sep_wide)

    if inp["sample_detections"]:
        print("\nSample Recognitions (qOCR):")
        for idx, sample in enumerate(inp["sample_detections"], start=1):
            print(
                f"  [{idx}] {sample['text']:<35} (score: {sample['score']:.3f}, angle: {sample['angle']}°)"
            )
        if inp["num_detections"] > len(inp["sample_detections"]):
            print(
                f"  ... and {inp['num_detections'] - len(inp['sample_detections'])} more"
            )

    if rapid_report and rapid_report.get("sample_detections"):
        print("\nSample Recognitions (RapidOCR):")
        for idx, sample in enumerate(rapid_report["sample_detections"], start=1):
            print(f"  [{idx}] {sample['text']:<35} (score: {sample['score']:.3f})")
        if rapid_report["num_detections"] > len(rapid_report["sample_detections"]):
            print(
                f"  ... and {rapid_report['num_detections'] - len(rapid_report['sample_detections'])} more"
            )
    print(sep_wide + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qocr-benchmark",
        description="Latency and throughput benchmark for qOCR vs RapidOCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--det-model-path",
        default=_default_path("models/det/det_model.onnx"),
        help="Path to detection ONNX model",
    )
    parser.add_argument(
        "--cls-model-path",
        default=_default_path("models/cls/cls_model.onnx"),
        help="Path to classification ONNX model",
    )
    parser.add_argument(
        "--rec-model-path",
        default=_default_path("models/rec/rec_model.onnx"),
        help="Path to recognition ONNX model",
    )
    parser.add_argument(
        "--dict-path",
        default=_default_path("models/ppocrv6_dict.txt"),
        help="Path to character dictionary file",
    )
    parser.add_argument(
        "--rapid-det-model-path",
        "--rapid-det-model",
        default=None,
        help="Path to RapidOCR detection ONNX model (defaults to --det-model-path)",
    )
    parser.add_argument(
        "--rapid-cls-model-path",
        "--rapid-cls-model",
        default=None,
        help="Path to RapidOCR classification ONNX model (defaults to --cls-model-path)",
    )
    parser.add_argument(
        "--rapid-rec-model-path",
        "--rapid-rec-model",
        default=None,
        help="Path to RapidOCR recognition ONNX model (defaults to --rec-model-path)",
    )
    parser.add_argument(
        "--rapid-dict-path",
        "--rapid-dict",
        default=None,
        help="Path to RapidOCR character dictionary (defaults to --dict-path)",
    )
    parser.add_argument(
        "-d", "--device-id", type=int, default=0, help="CUDA device index"
    )
    parser.add_argument(
        "-n", "--iterations", type=int, default=50, help="Benchmark iterations"
    )
    parser.add_argument("-w", "--warmup", type=int, default=10, help="Warm-up runs")
    parser.add_argument(
        "--no-breakdown", action="store_true", help="Skip per-stage profiling"
    )
    parser.add_argument("--no-disk", action="store_true", help="Skip disk timing")
    parser.add_argument(
        "--no-rapidocr", action="store_true", help="Skip RapidOCR comparison"
    )
    parser.add_argument(
        "--tensorrt",
        "--trt",
        dest="use_tensorrt",
        action="store_true",
        help="Enable TensorRT",
    )
    parser.add_argument(
        "--no-sort-width", action="store_true", help="Disable sorting by width"
    )
    parser.add_argument(
        "--min-text-score", type=float, default=0.5, help="Min confidence threshold"
    )
    parser.add_argument(
        "--no-drop-empty", action="store_true", help="Keep empty detections"
    )
    parser.add_argument(
        "--det-thresh", type=float, default=None, help="DBNet probability threshold"
    )
    parser.add_argument(
        "--det-box-thresh", type=float, default=None, help="DBNet box score threshold"
    )
    parser.add_argument(
        "--det-unclip-ratio",
        type=float,
        default=None,
        help="DBNet polygon unclip ratio",
    )
    parser.add_argument(
        "--json", dest="output_json", action="store_true", help="Output JSON"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = run_benchmark(
        image_path=args.image_path,
        det_model_path=args.det_model_path,
        rec_model_path=args.rec_model_path,
        cls_model_path=args.cls_model_path,
        dict_path=args.dict_path,
        device_id=args.device_id,
        iterations=args.iterations,
        warmup=args.warmup,
        profile_stages=not args.no_breakdown,
        include_disk=not args.no_disk,
        compare_rapidocr=not args.no_rapidocr,
        use_tensorrt=args.use_tensorrt,
        sort_by_width=not args.no_sort_width,
        min_text_score=args.min_text_score,
        drop_empty=not args.no_drop_empty,
        det_thresh=args.det_thresh,
        det_box_thresh=args.det_box_thresh,
        det_unclip_ratio=args.det_unclip_ratio,
        rapid_det_model_path=args.rapid_det_model_path,
        rapid_rec_model_path=args.rapid_rec_model_path,
        rapid_cls_model_path=args.rapid_cls_model_path,
        rapid_dict_path=args.rapid_dict_path,
    )

    if args.output_json:
        export_data = {k: v for k, v in report.items() if k != "stats_objects"}
        if "rapidocr" in export_data and isinstance(export_data["rapidocr"], dict):
            export_data["rapidocr"] = {
                k: v for k, v in export_data["rapidocr"].items() if k != "stats_objects"
            }
        print(json.dumps(export_data, indent=2))
    else:
        print_benchmark_table(report)


if __name__ == "__main__":
    main()
