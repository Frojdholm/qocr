import cupy as cp
import cv2
import numpy as np


def get_crops(
    image: np.ndarray | cp.ndarray,
    boxes: np.ndarray,
) -> list[np.ndarray]:
    """Lazy CPU crop extraction (for visualization / debugging)."""
    im_cpu = cp.asnumpy(image) if isinstance(image, cp.ndarray) else image
    crops = []
    for box in boxes:
        tl, tr, br, bl = box.astype(np.float32)
        crop_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        crop_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

        if crop_w <= 0 or crop_h <= 0:
            continue

        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array(
            [[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], dtype=np.float32
        )
        m = cv2.getPerspectiveTransform(src, dst)
        crops.append(cv2.warpPerspective(im_cpu, m, (crop_w, crop_h)))
    return crops


def render_boxes(
    image: np.ndarray | cp.ndarray,
    boxes: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Renders polygon bounding boxes onto an image canvas."""
    im_cpu = cp.asnumpy(image) if isinstance(image, cp.ndarray) else image
    canvas = im_cpu.copy()
    for box in boxes:
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
    return canvas


def render_heatmap(
    prob_map: np.ndarray | cp.ndarray,
    image: np.ndarray | cp.ndarray | None = None,
    target_size: tuple[int, int] | None = None,
    overlay: bool = False,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_PARULA,
) -> np.ndarray:
    """Renders probability map as a colormap heatmap, with optional image overlay."""
    prob_cpu = cp.asnumpy(prob_map) if isinstance(prob_map, cp.ndarray) else prob_map

    if target_size:
        prob_cpu = cv2.resize(prob_cpu, target_size, interpolation=cv2.INTER_LINEAR)
    elif image is not None and overlay:
        h, w = image.shape[:2]
        prob_cpu = cv2.resize(prob_cpu, (w, h), interpolation=cv2.INTER_LINEAR)

    u8 = (np.clip(prob_cpu, 0.0, 1.0) * 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(u8, colormap)
    hm_rgb = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)

    if overlay and image is not None:
        im_cpu = cp.asnumpy(image) if isinstance(image, cp.ndarray) else image
        return cv2.addWeighted(im_cpu, 1.0 - alpha, hm_rgb, alpha, 0.0)
    return hm_rgb
