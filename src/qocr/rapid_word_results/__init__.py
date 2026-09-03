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

from qocr.rapid_word_results.cal_rec_boxes import CalRecBoxes, Direction
from qocr.rapid_word_results.ctc_decode import (
    extract_word_info_from_ctc,
    get_word_info,
    has_chinese_char,
    is_chinese_char,
    quads_to_rect_bbox,
)
from qocr.rapid_word_results.main import calc_word_boxes, flatten_word_results
from qocr.rapid_word_results.word_info import (
    TextRecOutput,
    WordInfo,
    WordResult,
    WordType,
)

__all__ = [
    "CalRecBoxes",
    "Direction",
    "TextRecOutput",
    "WordInfo",
    "WordResult",
    "WordType",
    "calc_word_boxes",
    "extract_word_info_from_ctc",
    "flatten_word_results",
    "get_word_info",
    "has_chinese_char",
    "is_chinese_char",
    "quads_to_rect_bbox",
]
