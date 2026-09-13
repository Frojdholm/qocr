# Copyright (c) 2021-present RapidAI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qocr.rapid_to_markdown.to_markdown import ToMarkdown


def to_markdown(
    results_or_boxes: Any = None,
    txts: Sequence[str] | None = None,
) -> str:
    """Converts OCR results or bounding boxes and texts into layout-approximated Markdown.

    This function accepts either:
    1. A list/sequence of `OCRResult` objects (or any objects having `box` and `text` attributes):
       >>> to_markdown(results)

    2. Separate bounding boxes and text strings:
       >>> to_markdown(boxes, txts)

    3. A sequence of (box, text) pairs:
       >>> to_markdown([(box1, text1), (box2, text2)])

    Args:
        results_or_boxes: Sequence of OCRResult objects, list of boxes, or list of (box, text) tuples.
        txts: Optional sequence of recognized text strings when boxes are passed as the first argument.

    Returns:
        Formatted Markdown text preserving spatial layout.
    """
    if results_or_boxes is None:
        return ToMarkdown.to(None, txts)

    if txts is not None:
        return ToMarkdown.to(results_or_boxes, txts)

    if isinstance(results_or_boxes, (list, tuple)):
        if len(results_or_boxes) == 0:
            return ""

        first = results_or_boxes[0]
        if hasattr(first, "box") and hasattr(first, "text"):
            boxes = [r.box for r in results_or_boxes]
            texts = [str(r.text) for r in results_or_boxes]
            return ToMarkdown.to(boxes, texts)

        if (
            isinstance(first, (list, tuple))
            and len(first) == 2
            and isinstance(first[1], str)
        ):
            boxes = [item[0] for item in results_or_boxes]
            texts = [str(item[1]) for item in results_or_boxes]
            return ToMarkdown.to(boxes, texts)

    return ToMarkdown.to(results_or_boxes, None)
