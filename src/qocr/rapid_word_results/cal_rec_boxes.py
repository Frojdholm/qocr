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

import copy
import math
from enum import Enum
from typing import Any

import cv2
import numpy as np

from qocr.rapid_word_results.ctc_decode import quads_to_rect_bbox
from qocr.rapid_word_results.word_info import (
    TextRecOutput,
    WordInfo,
    WordType,
)


class Direction(Enum):
    HORIZONTAL = "horizontal_direct"  # 水平
    VERTICAL = "vertical_direct"  # 垂直


class CalRecBoxes:
    """计算识别文字的汉字单字和英文单词的坐标框。
    代码借鉴自PaddlePaddle/PaddleOCR和fanqie03/char-detection以及RapidOCR。"""

    def __call__(
        self,
        imgs: list[np.ndarray] | None,
        dt_boxes: np.ndarray | list[np.ndarray],
        rec_res: TextRecOutput | Any,
        return_single_char_box: bool = False,
    ) -> TextRecOutput | Any:
        word_results = []
        dt_boxes_list = list(dt_boxes)
        num_boxes = len(dt_boxes_list)

        for idx in range(num_boxes):
            box = np.asarray(dt_boxes_list[idx], dtype=np.float32)

            txts = getattr(rec_res, "txts", None)
            res_word_results = getattr(rec_res, "word_results", None)

            if (
                txts is None
                or idx >= len(txts)
                or res_word_results is None
                or idx >= len(res_word_results)
            ):
                word_results.append([])
                continue

            img = imgs[idx] if imgs is not None and idx < len(imgs) else None
            direction = self.get_box_direction(box)

            if img is not None and hasattr(img, "shape") and img.size > 0:
                h, w = img.shape[:2]
            else:
                img_crop_width = int(
                    max(
                        float(np.linalg.norm(box[0] - box[1])),
                        float(np.linalg.norm(box[2] - box[3])),
                    )
                )
                img_crop_height = int(
                    max(
                        float(np.linalg.norm(box[0] - box[3])),
                        float(np.linalg.norm(box[1] - box[2])),
                    )
                )
                if direction == Direction.VERTICAL:
                    w, h = max(img_crop_height, 1), max(img_crop_width, 1)
                else:
                    w, h = max(img_crop_width, 1), max(img_crop_height, 1)

            img_box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            word_box_content_list, word_box_list, conf_list = self.cal_ocr_word_box(
                txts[idx],
                img_box,
                res_word_results[idx],
                return_single_char_box,
            )
            word_box_list = self.adjust_box_overlap(copy.deepcopy(word_box_list))
            word_box_list = self.reverse_rotate_crop_image(
                copy.deepcopy(box), word_box_list, direction
            )
            word_results.append(
                list(zip(word_box_content_list, conf_list, word_box_list))
            )

        if hasattr(rec_res, "word_results"):
            rec_res.word_results = tuple(word_results)
        return rec_res

    @staticmethod
    def get_box_direction(box: np.ndarray) -> Direction:
        edge_lengths = [
            float(np.linalg.norm(box[0] - box[1])),  # 上边
            float(np.linalg.norm(box[1] - box[2])),  # 右边
            float(np.linalg.norm(box[2] - box[3])),  # 下边
            float(np.linalg.norm(box[3] - box[0])),  # 左边
        ]

        # 宽和高取对边的最大距离
        width = max(edge_lengths[0], edge_lengths[2])
        height = max(edge_lengths[1], edge_lengths[3])

        if width < 1e-6:
            return Direction.VERTICAL

        aspect_ratio = round(height / width, 2)
        return Direction.VERTICAL if aspect_ratio >= 1.5 else Direction.HORIZONTAL

    def cal_ocr_word_box(
        self,
        rec_txt: str,
        bbox: np.ndarray,
        word_info: WordInfo,
        return_single_char_box: bool = False,
    ) -> tuple[list[str], list[list[list[int]]], list[float]]:
        """Calculate the detection frame for each word based on the results of recognition and detection of ocr.

        汉字坐标是单字的
        英语坐标是单词级别的
        三种情况：
        1. 全是汉字
        2. 全是英文
        3. 中英混合
        """
        if not rec_txt or word_info.line_txt_len == 0:
            return [], [], []

        bbox_points = quads_to_rect_bbox(bbox[None, ...])
        avg_col_width = (bbox_points[2] - bbox_points[0]) / word_info.line_txt_len

        is_all_en_num = all(v is WordType.EN_NUM for v in word_info.word_types)

        line_cols, char_widths, word_contents = [], [], []
        word_confs = []
        conf_idx = 0

        for word, word_col in zip(word_info.words, word_info.word_cols):
            word_len = len(word)
            cur_confs = (
                word_info.confs[conf_idx : conf_idx + word_len]
                if word_info.confs
                else []
            )
            conf_idx += word_len

            if is_all_en_num and not return_single_char_box:
                line_cols.append(word_col)
                word_contents.append("".join(word))
                word_confs.append(float(np.mean(cur_confs)) if cur_confs else 1.0)
            else:
                line_cols.extend(word_col)
                word_contents.extend(word)
                if cur_confs:
                    word_confs.extend(cur_confs)
                else:
                    word_confs.extend([1.0] * word_len)

            if len(word_col) == 1:
                continue

            avg_width = self.calc_avg_char_width(word_col, avg_col_width)
            char_widths.append(avg_width)

        avg_char_width = self.calc_all_char_avg_width(
            char_widths, bbox_points[0], bbox_points[2], len(rec_txt)
        )

        if is_all_en_num and not return_single_char_box:
            word_boxes = self.calc_en_num_box(
                line_cols, avg_char_width, avg_col_width, bbox_points
            )
        else:
            word_boxes = self.calc_box(
                line_cols, avg_char_width, avg_col_width, bbox_points
            )
        return word_contents, word_boxes, word_confs

    def calc_en_num_box(
        self,
        line_cols: list[list[int]],
        avg_char_width: float,
        avg_col_width: float,
        bbox_points: tuple[float, float, float, float],
    ) -> list[list[list[int]]]:
        results = []
        for one_col in line_cols:
            cur_word_cell = self.calc_box(
                one_col, avg_char_width, avg_col_width, bbox_points
            )
            if not cur_word_cell:
                continue
            x0, y0, x1, y1 = quads_to_rect_bbox(np.array(cur_word_cell))
            results.append(
                [
                    [int(x0), int(y0)],
                    [int(x1), int(y0)],
                    [int(x1), int(y1)],
                    [int(x0), int(y1)],
                ]
            )
        return results

    @staticmethod
    def calc_box(
        line_cols: list[int],
        avg_char_width: float,
        avg_col_width: float,
        bbox_points: tuple[float, float, float, float],
    ) -> list[list[list[int]]]:
        x0, y0, x1, y1 = bbox_points

        results = []
        for col_idx in line_cols:
            # 将中心点定位在列的中间位置
            center_x = (col_idx + 0.5) * avg_col_width

            # 计算字符单元格的左右边界
            char_x0 = max(int(center_x - avg_char_width / 2), 0) + x0
            char_x1 = min(int(center_x + avg_char_width / 2), x1 - x0) + x0
            cell = [
                [int(char_x0), int(y0)],
                [int(char_x1), int(y0)],
                [int(char_x1), int(y1)],
                [int(char_x0), int(y1)],
            ]
            results.append(cell)
        return sorted(results, key=lambda x: x[0][0])

    @staticmethod
    def calc_avg_char_width(word_col: list[int], each_col_width: float) -> float:
        char_total_length = (word_col[-1] - word_col[0]) * each_col_width
        return char_total_length / (len(word_col) - 1)

    @staticmethod
    def calc_all_char_avg_width(
        width_list: list[float], bbox_x0: float, bbox_x1: float, txt_len: int
    ) -> float:
        if txt_len == 0:
            return 0.0

        if len(width_list) > 0:
            return float(sum(width_list) / len(width_list))

        return float((bbox_x1 - bbox_x0) / txt_len)

    @staticmethod
    def adjust_box_overlap(
        word_box_list: list[list[list[int]]],
    ) -> list[list[list[int]]]:
        # 调整bbox有重叠的地方
        for i in range(len(word_box_list) - 1):
            cur, nxt = word_box_list[i], word_box_list[i + 1]
            if cur[1][0] > nxt[0][0]:  # 有交集
                distance = abs(cur[1][0] - nxt[0][0])
                cur[1][0] -= int(distance / 2)
                cur[2][0] -= int(distance / 2)
                nxt[0][0] += int(distance - distance / 2)
                nxt[3][0] += int(distance - distance / 2)
        return word_box_list

    def reverse_rotate_crop_image(
        self,
        bbox_points: np.ndarray,
        word_points_list: list[list[list[int]]],
        direction: Direction,
    ) -> list[list[list[int]]]:
        """get_rotate_crop_image的逆操作

        img为原图
        part_img为crop后的图
        bbox_points为part_img中对应在原图的bbox, 四个点，左上，右上，右下，左下
        part_points为在part_img中的点[(x, y), (x, y)]
        """
        bbox_points = bbox_points.astype(np.float32)
        left = int(np.min(bbox_points[:, 0]))
        top = int(np.min(bbox_points[:, 1]))
        bbox_points[:, 0] = bbox_points[:, 0] - left
        bbox_points[:, 1] = bbox_points[:, 1] - top

        img_crop_width = int(np.linalg.norm(bbox_points[0] - bbox_points[1]))
        img_crop_height = int(np.linalg.norm(bbox_points[0] - bbox_points[3]))

        pts_std = np.array(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        ).astype(np.float32)
        M = cv2.getPerspectiveTransform(bbox_points, pts_std)
        _, IM = cv2.invert(M)

        new_word_points_list = []
        for word_points in word_points_list:
            new_word_points = []
            for point in word_points:
                new_point = point
                if direction == Direction.VERTICAL:
                    new_point = self.s_rotate(
                        math.radians(-90), new_point[0], new_point[1], 0, 0
                    )
                    new_point[0] = new_point[0] + img_crop_width

                p = np.array(list(new_point) + [1])
                x, y, z = np.dot(IM, p)
                new_point = [x / z, y / z]

                new_point = [int(new_point[0] + left), int(new_point[1] + top)]
                new_word_points.append(new_point)
            new_word_points = self.order_points(new_word_points)
            new_word_points_list.append(new_word_points)
        return new_word_points_list

    @staticmethod
    def s_rotate(
        angle: float, valuex: float, valuey: float, pointx: float, pointy: float
    ) -> list[float]:
        """绕pointx,pointy顺时针旋转"""
        s_rotate_x = (
            (valuex - pointx) * math.cos(angle)
            + (valuey - pointy) * math.sin(angle)
            + pointx
        )
        s_rotate_y = (
            (valuey - pointy) * math.cos(angle)
            - (valuex - pointx) * math.sin(angle)
            + pointy
        )
        return [float(s_rotate_x), float(s_rotate_y)]

    @staticmethod
    def order_points(ori_box: list[list[int]]) -> list[list[int]]:
        """矩形框顺序排列"""

        def convert_to_1x2(p):
            if p.shape == (2,):
                return p.reshape((1, 2))
            if p.shape == (1, 2):
                return p
            return p[:1, :]

        box = np.array(ori_box).reshape((-1, 2))
        center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
        if np.any(box[:, 0] == center_x) and np.any(box[:, 1] == center_y):
            p1 = box[np.where(box[:, 0] == np.min(box[:, 0]))]
            p2 = box[np.where(box[:, 1] == np.min(box[:, 1]))]
            p3 = box[np.where(box[:, 0] == np.max(box[:, 0]))]
            p4 = box[np.where(box[:, 1] == np.max(box[:, 1]))]
        elif np.all(box[:, 0] == center_x):
            y_sort = np.argsort(box[:, 1])
            p1 = box[y_sort[0]]
            p2 = box[y_sort[1]]
            p3 = box[y_sort[2]]
            p4 = box[y_sort[3]]
        elif np.any(box[:, 0] == center_x) and np.all(box[:, 1] != center_y):
            p12, p34 = (
                box[np.where(box[:, 1] < center_y)],
                box[np.where(box[:, 1] > center_y)],
            )
            p1, p2 = (
                p12[np.where(p12[:, 0] == np.min(p12[:, 0]))],
                p12[np.where(p12[:, 0] == np.max(p12[:, 0]))],
            )
            p3, p4 = (
                p34[np.where(p34[:, 0] == np.max(p34[:, 0]))],
                p34[np.where(p34[:, 0] == np.min(p34[:, 0]))],
            )
        else:
            p14, p23 = (
                box[np.where(box[:, 0] < center_x)],
                box[np.where(box[:, 0] > center_x)],
            )
            p1, p4 = (
                p14[np.where(p14[:, 1] == np.min(p14[:, 1]))],
                p14[np.where(p14[:, 1] == np.max(p14[:, 1]))],
            )
            p2, p3 = (
                p23[np.where(p23[:, 1] == np.min(p23[:, 1]))],
                p23[np.where(p23[:, 1] == np.max(p23[:, 1]))],
            )

        p1 = convert_to_1x2(p1)
        p2 = convert_to_1x2(p2)
        p3 = convert_to_1x2(p3)
        p4 = convert_to_1x2(p4)
        return np.array([p1, p2, p3, p4]).reshape((-1, 2)).tolist()
