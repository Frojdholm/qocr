from __future__ import annotations

import pathlib

from qocr.pipeline import OCRPipeline, OCRResult


def convert(
    image_path: pathlib.Path | str,
    det_model_path: pathlib.Path | str,
    cls_model_path: pathlib.Path | str | None,
    rec_model_path: pathlib.Path | str,
    dict_path: pathlib.Path | str | None = None,
    visualize: bool = True,
    return_word_box: bool = False,
    return_single_char_box: bool = False,
) -> list[OCRResult]:
    import cv2

    pipeline = OCRPipeline(
        det_model_path=det_model_path,
        cls_model_path=cls_model_path,
        rec_model_path=rec_model_path,
        dict_path=dict_path,
        return_word_box=return_word_box,
        return_single_char_box=return_single_char_box,
    )

    im_gpu = pipeline._prepare_gpu_image(image_path)

    results, prob_map = pipeline.predict(
        im_gpu,
        return_word_box=return_word_box,
        return_single_char_box=return_single_char_box,
    )

    for r in results:
        print(f"[Orientation] {r.angle}° ({r.angle_score:.3f})")
        print(f"[Text {r.score:.3f}] {r.text}")
        if r.word_results:
            words_desc = ", ".join(f"'{w.text}'({w.score:.3f})" for w in r.word_results)
            print(f"  [Words] {words_desc}")

    if visualize and prob_map is not None:
        import cupy as cp
        import numpy as np

        from qocr.cls import debug as cls_debug
        from qocr.det import debug as det_debug
        from qocr.rec import debug as rec_debug

        boxes = (
            np.array([r.box for r in results], dtype=np.float32)
            if results
            else np.empty((0, 4, 2), dtype=np.float32)
        )
        im_cpu = cv2.imread(str(image_path), cv2.IMREAD_COLOR_RGB)
        prob_cpu = cp.asnumpy(prob_map)

        cls_vis = cls_debug.render_orientation(im_cpu, boxes, results)
        res_vis = rec_debug.render_results(im_cpu, boxes, results)
        boxes_vis = det_debug.render_boxes(im_cpu, boxes)
        heatmap_vis = det_debug.render_heatmap(prob_cpu, im_cpu, overlay=True)

        cv2.imshow("boxes", cv2.cvtColor(boxes_vis, cv2.COLOR_RGB2BGR))
        cv2.imshow("heatmap", cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR))
        cv2.imshow("orientation", cv2.cvtColor(cls_vis, cv2.COLOR_RGB2BGR))
        cv2.imshow("recognition", cv2.cvtColor(res_vis, cv2.COLOR_RGB2BGR))

        if return_word_box and any(r.word_results for r in results):
            words_vis = rec_debug.render_word_results(im_cpu, results)
            cv2.imshow("word_results", cv2.cvtColor(words_vis, cv2.COLOR_RGB2BGR))

        cv2.waitKey(0)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_path", help="Image to run OCR on")
    parser.add_argument(
        "--det-model-path",
        dest="det_model_path",
        help="Path to text detection ONNX model file (.onnx)",
        required=True,
    )
    parser.add_argument(
        "--cls-model-path",
        dest="cls_model_path",
        help="Path to textline orientation classification ONNX model file (.onnx)",
        default=None,
    )
    parser.add_argument(
        "--rec-model-path",
        dest="rec_model_path",
        help="Path to text recognition ONNX model file (.onnx)",
        required=True,
    )
    parser.add_argument(
        "--dict-path",
        dest="dict_path",
        help="Path to character dictionary file (.txt or .yaml)",
        required=True,
    )
    parser.add_argument(
        "--return-word-box",
        dest="return_word_box",
        action="store_true",
        help="Whether to extract word-level bounding boxes and confidences",
    )
    parser.add_argument(
        "--return-single-char-box",
        dest="return_single_char_box",
        action="store_true",
        help="Whether to return individual character-level boxes",
    )

    args = parser.parse_args()

    convert(
        image_path=pathlib.Path(args.image_path),
        det_model_path=pathlib.Path(args.det_model_path),
        cls_model_path=pathlib.Path(args.cls_model_path)
        if args.cls_model_path
        else None,
        rec_model_path=pathlib.Path(args.rec_model_path),
        dict_path=pathlib.Path(args.dict_path) if args.dict_path else None,
        return_word_box=args.return_word_box,
        return_single_char_box=args.return_single_char_box,
    )
