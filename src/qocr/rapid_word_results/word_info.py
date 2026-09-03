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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

import numpy as np


class WordType(Enum):
    CN = "cn"
    EN = "en"
    NUM = "num"
    EN_NUM = "en&num"


@dataclass
class WordInfo:
    words: list[list[str]] = field(default_factory=list)
    word_cols: list[list[int]] = field(default_factory=list)
    word_types: list[WordType] = field(default_factory=list)
    line_txt_len: float = 0.0
    confs: list[float] = field(default_factory=list)


class WordResult(NamedTuple):
    text: str
    score: float
    box: np.ndarray

    def __str__(self) -> str:
        return f"{self.text} ({self.score:.4f})"


@dataclass
class TextRecOutput:
    imgs: list[np.ndarray] | None = None
    txts: tuple[str, ...] | None = None
    scores: list[float] = field(default_factory=lambda: [1.0])
    word_results: tuple[Any, ...] = field(default_factory=tuple)
    elapse: float | None = None

    def __len__(self) -> int:
        if self.txts is None:
            return 0
        return len(self.txts)
