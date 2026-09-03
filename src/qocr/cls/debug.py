from __future__ import annotations

import cupy as cp
import cv2
import numpy as np

from qocr.pipeline import OCRResult


def render_orientation(
    image: np.ndarray | cp.ndarray,
    boxes: np.ndarray,
    results: list[OCRResult],
    color_0: tuple[int, int, int] = (0, 255, 0),
    color_180: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Renders polygon bounding boxes colored according to orientation angle (0° vs 180°)."""
    im_cpu = cp.asnumpy(image) if isinstance(image, cp.ndarray) else image
    canvas = im_cpu.copy()

    for box, res in zip(boxes, results):
        angle = res.angle
        score = res.angle_score

        color = color_180 if angle == 180 else color_0
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

        label = f"{angle}° ({score:.2f})"
        tl = box[0].astype(int)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        text_y = max(tl[1], th + 4)
        bg_tl = (tl[0], text_y - th - 4)
        bg_br = (tl[0] + tw + 4, text_y + baseline)
        cv2.rectangle(canvas, bg_tl, bg_br, color, -1)
        cv2.putText(
            canvas,
            label,
            (tl[0] + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return canvas
