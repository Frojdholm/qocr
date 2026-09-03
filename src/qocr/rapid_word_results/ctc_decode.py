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

import numpy as np

from qocr.rapid_word_results.word_info import WordInfo, WordType


def is_chinese_char(ch: str) -> bool:
    return (
        "\u4e00" <= ch <= "\u9fff"  # 汉字
        or "\u3000" <= ch <= "\u303f"  # CJK 标点（如 。 、 “” 《》 ……）
        or "\uff00" <= ch <= "\uffef"  # 全角符号（如 ，．！？【】）
    )


def has_chinese_char(text: str) -> bool:
    return any(is_chinese_char(ch) for ch in text)


def quads_to_rect_bbox(bbox: np.ndarray) -> tuple[float, float, float, float]:
    """Converts (N, 4, 2) or (4, 2) quadrilateral box array into enclosing (x_min, y_min, x_max, y_max)."""
    if bbox.ndim == 2:
        bbox = bbox[None, ...]

    if bbox.ndim != 3:
        raise ValueError("bbox shape must be 3 (or 2)")

    if bbox.shape[1] != 4 or bbox.shape[2] != 2:
        raise ValueError("bbox shape must be (N, 4, 2)")

    all_x, all_y = (bbox[:, :, 0].flatten(), bbox[:, :, 1].flatten())
    x_min, y_min = np.min(all_x), np.min(all_y)
    x_max, y_max = np.max(all_x), np.max(all_y)
    return float(x_min), float(y_min), float(x_max), float(y_max)


def get_word_info(text: str, selection: np.ndarray) -> WordInfo:
    """Group the decoded characters and record the corresponding decoded positions.

    Based on PaddlePaddle/PaddleOCR rec_postprocess.py and RapidOCR ch_ppocr_rec/utils.py.
    """
    word_list = []
    word_col_list = []
    state_list = []

    word_content = []
    word_col_content = []

    valid_col = np.where(selection)[0]
    if len(valid_col) <= 0 or not text:
        return WordInfo()

    col_width = np.zeros(valid_col.shape)
    col_width[1:] = valid_col[1:] - valid_col[:-1]
    col_width[0] = min(3 if has_chinese_char(text[0]) else 2, int(valid_col[0]))

    state = None
    for c_i, char in enumerate(text):
        if char.isspace():
            if word_content:
                word_list.append(word_content)
                word_col_list.append(word_col_content)
                state_list.append(state)
                word_content = []
                word_col_content = []
            continue

        c_state = WordType.CN if has_chinese_char(char) else WordType.EN_NUM
        if state is None:
            state = c_state

        if state != c_state or col_width[c_i] > 5:
            if len(word_content) != 0:
                word_list.append(word_content)
                word_col_list.append(word_col_content)
                state_list.append(state)
                word_content = []
                word_col_content = []
            state = c_state

        word_content.append(char)
        word_col_content.append(int(valid_col[c_i]))

    if len(word_content) != 0:
        word_list.append(word_content)
        word_col_list.append(word_col_content)
        state_list.append(state)

    return WordInfo(words=word_list, word_cols=word_col_list, word_types=state_list)


def extract_word_info_from_ctc(
    tokens: np.ndarray,
    probs: np.ndarray | None,
    char_dict: list[str],
    wh_ratio: float = 1.0,
    max_wh_ratio: float = 1.0,
    blank_idx: int = 0,
) -> tuple[str, float, WordInfo]:
    """Decodes CTC sequence output and extracts text, line score, and WordInfo."""
    seq_len = len(tokens)
    selection = np.ones(seq_len, dtype=bool)
    selection[1:] = tokens[1:] != tokens[:-1]
    selection &= tokens != blank_idx
    selection &= tokens < len(char_dict)

    if probs is not None:
        conf_list = [round(float(conf), 5) for conf in probs[selection]]
    else:
        conf_list = [1.0] * int(np.sum(selection))

    if not conf_list:
        conf_list = [0.0]

    chars = [char_dict[tid] for tid in tokens[selection]]
    text = "".join(chars)
    line_score = float(np.mean(conf_list)) if conf_list else 0.0

    word_info = get_word_info(text, selection)
    effective_ratio = (wh_ratio / max_wh_ratio) if max_wh_ratio > 0 else 1.0
    word_info.line_txt_len = seq_len * effective_ratio
    word_info.confs = conf_list

    return text, line_score, word_info
