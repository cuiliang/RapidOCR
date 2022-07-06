# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import base64
import copy
import json
from functools import reduce
from pathlib import Path

import cv2
import numpy as np

from rapidocr_onnxruntime import TextSystem

text_sys = TextSystem('config.yaml')


def detect_recognize(image_path):
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        raise TypeError(f'{image_path} is not str or ndarray.')

    dt_boxes, rec_res, img, elapse_part = text_sys(image)

    if dt_boxes is None or rec_res is None:
        #rec_res_data = json.dumps([],
        #                          indent=2,
        #                          ensure_ascii=False)
        temp_rec_res = []
        rec_res_data = temp_rec_res
        elapse = 0
        elapse_part = ''
        # image = cv2.imencode('.jpg', img)[1]
        # img = str(base64.b64encode(image))[2:-1]
    else:
        temp_rec_res = []
        for i, value in enumerate(rec_res):
            temp_rec_res.append([i, value[0], value[1]])
        temp_rec_res = np.array(temp_rec_res)

        # rec_res_data = json.dumps(temp_rec_res.tolist(),
        #                          indent=2,
        #                          ensure_ascii=False)

        rec_res_data = temp_rec_res.tolist()

        # det_im = draw_text_det_res(dt_boxes, img)
        # image = cv2.imencode('.jpg', det_im)[1]
        # img = str(base64.b64encode(image))[2:-1]

        elapse = reduce(lambda x, y: float(x)+float(y), elapse_part)
        # elapse_part = ','.join([str(x) for x in elapse_part])
   
    #return json.dumps({'image': img,
    #                   'total_elapse': f'{elapse:.4f}',
    #                   'elapse_part': elapse_part,
    #                   'rec_res': rec_res_data})

        #elapse = sum(
        #        [
        #            *elapse_part.values()
        #        ]
        #    )

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
                    "left": round(rect[0][0]),
                    "top": round(rect[0][1]),
                    "right": round(rect[1][0]),
                    "bottom": round(rect[1][1]),
                },
            }
        )

    
    lines = ""
    _last_top = 0
    templine = []  # 同一行的，先放在一起，然后根据left排序一下再输出。比较因为较少的高度差导致顺序错位
    for region in regions:
        this_top = region["rect"]["top"]
        if abs(this_top - _last_top) > 10:       # 发生了换行
            if len(templine) > 0:
                templine.sort(key = lambda x: x["rect"]["left"])
                for block in templine:
                    lines += " " + block["text"]
                lines += "\n"
            if (this_top - _last_top > 60): # 多个换行
               lines += "\n"


            # 开启新行
            templine.clear()
            templine.append(region)

            #lines += "\n"
            #if (this_top - _last_top > 60): # 多个换行
            #    lines += "\n"
            #lines += region["text"]
        else:
            templine.append(region)
            # lines += " "
            # lines += region["text"]

        _last_top = this_top
    
    # 还没有输出的内容输出一下
    if len(templine) > 0:
        templine.sort(key = lambda x: x["rect"]["left"])
        for block in templine:
            lines += " " + block["text"]
        lines += "\n"
    if (this_top - _last_top > 60): # 多个换行
        lines += "\n"

    lines = lines.lstrip()

    #####################################    
    # lines = ""
    # _last_top = 0
    # templine = []  # 同一行的，先放在一起，然后根据left排序一下再输出。比较因为较少的高度差导致顺序错位
    # for region in regions:
    #     this_top = region["rect"]["top"]
    #     if this_top - _last_top > 15:       # 发生了换行
    #         lines += "\n"
    #         if (this_top - _last_top > 60): # 多个换行
    #             lines += "\n"
    #         lines += region["text"]
    #     else:
    #         lines += " "
    #         lines += region["text"]

    #     _last_top = this_top
    
    # lines = lines.lstrip()
    
    return json.dumps({
        # 'image': img,
        'result': {
            "lines": lines,
            "regions": regions,
        },
        'info': {
            'total_elapse': elapse,
            'elapse_part': {
                'det_elapse':elapse_part[0],
                'cls_elapse':elapse_part[1],
                'rec_elapse':elapse_part[2]
            },
        },
    })


def check_and_read_gif(img_path):
    if Path(img_path).name[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            print("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True
    return None, False


def draw_text_det_res(dt_boxes, raw_im):
    src_im = copy.deepcopy(raw_im)
    for i, box in enumerate(dt_boxes):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True,
                      color=(0, 0, 255),
                      thickness=1)
        cv2.putText(src_im, str(i), (int(box[0][0]), int(box[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return src_im
