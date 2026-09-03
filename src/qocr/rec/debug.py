from __future__ import annotations

import cupy as cp
import cv2
import numpy as np

from qocr.pipeline import OCRResult


def render_results(
    image: np.ndarray | cp.ndarray,
    boxes: np.ndarray,
    results: list[OCRResult],
    color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Renders polygon bounding boxes and recognized text labels onto an image canvas."""
    im_cpu = cp.asnumpy(image) if isinstance(image, cp.ndarray) else image
    canvas = im_cpu.copy()

    for box, res in zip(boxes, results):
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

        label = res.text
        if not label:
            continue

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
            text_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return canvas


def render_word_results(
    image: np.ndarray | cp.ndarray,
    results: list[OCRResult],
    box_color: tuple[int, int, int] = (255, 140, 0),
    line_color: tuple[int, int, int] = (0, 200, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: float = 0.4,
) -> np.ndarray:
    """Renders line polygons and individual word bounding boxes with labels onto an image canvas."""
    im_cpu = cp.asnumpy(image) if isinstance(image, cp.ndarray) else image
    canvas = im_cpu.copy()

    for res in results:
        line_pts = np.array(res.box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [line_pts], isClosed=True, color=line_color, thickness=1)

        if not res.word_results:
            continue

        for word in res.word_results:
            w_pts = np.array(word.box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                canvas, [w_pts], isClosed=True, color=box_color, thickness=thickness
            )

            label = word.text
            if not label:
                continue

            tl = word.box[0].astype(int)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            text_y = max(tl[1], th + 4)
            bg_tl = (tl[0], text_y - th - 4)
            bg_br = (tl[0] + tw + 2, text_y + baseline)
            cv2.rectangle(canvas, bg_tl, bg_br, box_color, -1)
            cv2.putText(
                canvas,
                label,
                (tl[0] + 1, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return canvas
