# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any

import numpy as np

from qocr.rapid_word_results.cal_rec_boxes import CalRecBoxes
from qocr.rapid_word_results.word_info import TextRecOutput, WordResult


def calc_word_boxes(
    imgs: list[np.ndarray] | None,
    dt_boxes: np.ndarray | list[np.ndarray],
    rec_res: TextRecOutput | Any,
    return_single_char_box: bool = False,
    cal_rec_boxes_op: CalRecBoxes | None = None,
) -> tuple[tuple[WordResult, ...], ...]:
    """Calculates word-level bounding boxes and wraps them as WordResult tuples."""
    if cal_rec_boxes_op is None:
        cal_rec_boxes_op = CalRecBoxes()

    cal_res = cal_rec_boxes_op(
        imgs=imgs,
        dt_boxes=dt_boxes,
        rec_res=rec_res,
        return_single_char_box=return_single_char_box,
    )

    raw_word_results = getattr(cal_res, "word_results", ())
    origin_words: list[tuple[WordResult, ...]] = []

    for word_line in raw_word_results:
        if not word_line:
            origin_words.append(())
            continue

        line_items: list[WordResult] = []
        for item in word_line:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                txt, score, bbox = item
                if bbox is not None:
                    line_items.append(
                        WordResult(
                            text=str(txt),
                            score=float(score),
                            box=np.asarray(bbox, dtype=np.int32),
                        )
                    )
            elif isinstance(item, WordResult):
                line_items.append(item)

        origin_words.append(tuple(line_items))

    return tuple(origin_words)


def flatten_word_results(word_results: Any) -> list[WordResult]:
    """Flattens nested word results from lines into a single flat list of WordResult items."""
    if not word_results:
        return []

    flattened: list[WordResult] = []
    for word_line in word_results:
        if not word_line:
            continue

        for word_item in word_line:
            if isinstance(word_item, WordResult):
                flattened.append(word_item)
            elif isinstance(word_item, (list, tuple)) and len(word_item) == 3:
                txt, score, bbox = word_item
                if bbox is not None:
                    flattened.append(
                        WordResult(
                            text=str(txt),
                            score=float(score),
                            box=np.asarray(bbox, dtype=np.int32),
                        )
                    )

    return flattened
