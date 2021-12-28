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

        try:
            dt_boxes = [i.tolist() for i in dt_boxes]
        except AttributeError:
            dt_boxes = dt_boxes

    regions = []
    for rect, res in zip(dt_boxes, rec_res_data):
        regions.append(
            {
                "text": res[1],
                "confidence": float(res[2]),
                "rect": {
                    "left": rect[0][0],
                    "top": rect[0][1],
                    "right": rect[1][0],
                    "bottom": rect[1][1],
                },
            }
        )
        
    lines = ""
    _last_top = 0
    for region in regions:
        this_top = region["rect"]["top"]
        if this_top - _last_top > 15:       # 发生了换行
            lines += "\n"
            if (this_top - _last_top > 50): # 多个换行
                lines += "\n"
            lines += region["text"]
        else:
            lines += " "
            lines += region["text"]

        _last_top = this_top
    
    lines = lines.lstrip()
    
    return json.dumps({
        # 'image': img,
        'result': {
            "lines": lines,
            "regions": regions,
        },
        'info': {
            'total_elapse': elapse,
            'elapse_part': elapse_part,
        },
    })


