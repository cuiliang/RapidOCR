# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: task.py
# @Time: 2021/03/07 20:29:32
# @Author: SWHL
# @Contact: liekkaskono@163.com
import base64
from functools import reduce
import json
import time

import cv2
import numpy as np

from resources.rapidOCR import TextSystem, draw_text_det_res, check_and_read_gif

det_model_path = 'resources/models/ch_PP-OCRv2_det_infer.onnx'
cls_model_path = 'resources/models/ch_ppocr_mobile_v2.0_cls_infer.onnx'
rec_model_path = 'resources/models/ch_ppocr_mobile_v2.0_rec_infer.onnx'

text_sys = TextSystem(
    det_model_path,
    rec_model_path,
    use_angle_cls=True,
    cls_model_path=cls_model_path
)


def detect_recognize(image_path):
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        raise TypeError(f'{image_path} is not str or ndarray.')

    dt_boxes, rec_res, img, elapse_part = text_sys(image)

    if dt_boxes is None or rec_res is None:
        temp_rec_res = []
        rec_res_data = temp_rec_res
        elapse = 0
        elapse_part = ''
        image = cv2.imencode('.jpg', img)[1]
        img = str(base64.b64encode(image))[2:-1]
    else:
        temp_rec_res = []
        for i, value in enumerate(rec_res):
            temp_rec_res.append([i, value[0], value[1]])
        temp_rec_res = np.array(temp_rec_res)

        rec_res_data = temp_rec_res.tolist()

        # det_im = draw_text_det_res(dt_boxes, img)
        # image = cv2.imencode('.jpg', det_im)[1]
        # img = str(base64.b64encode(image))[2:-1]

        # elapse = reduce(lambda x, y: float(x)+float(y), elapse_part)
        # elapse_part = ','.join([str(x) for x in elapse_part])

        elapse = sum(
            [
                *elapse_part.values()
            ]
        )

        dt_boxes = [i.tolist() for i in dt_boxes]

    return json.dumps({
        # 'image': img,
        'result': {
            "dt_boxes": dt_boxes,
            "rec_res_data": rec_res_data,
        },
        'info': {
            'total_elapse': elapse,
            'elapse_part': elapse_part,
        },
    })
