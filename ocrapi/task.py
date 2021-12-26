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

# 合并位置和结果列表
def merge_list(dt_boxes, dt_data):
    
    # 获得某个块的Rect坐标，left,top,right,height. pos_list: [[x,y],[x,y],[x,y],[x,y]]
    def get_rect(pos_list):
        #print('pos_list',pos_list)

        left = pos_list[0][0]
        right =  pos_list[0][0]
        top =  pos_list[0][1]
        bottom =  pos_list[0][1]
        for pos in pos_list:
            if pos[0] < left:
                left = pos[0]
            if pos[0] > right:
                right = pos[0] 
            if pos[1] < top:
                top = pos[1]
            if pos[1] > bottom:
                bottom = pos[1]
        return dict(left = left, top = top, right = right ,bottom = bottom)

    list = []
    for i in range(len(dt_boxes)):
        list.append(dict(rect = get_rect(dt_boxes[i]), text=dt_data[i][1]))
    return list

# 获取合并后的文本列表
from io import StringIO

def get_lines(region_list):
    #print('region_list[0]',region_list[0])
    lines = StringIO()
    top = region_list[0]['rect']['top']
    delta = (region_list[0]['rect']['bottom'] - region_list[0]['rect']['top']) * 0.7  # 判断是新行的高度
    two_line = delta * 4 # 空两行的高度
    start_new_line = True # 是否开始了一个新的行

    for region in region_list:
        item_top = region['rect']['top']
        item_text = region['text']
        if top > item_top - delta and top < item_top + delta: # 在同一行中
            if start_new_line:
                lines.write(item_text)
                start_new_line = False
            else:
                lines.write('  ')
                lines.write(item_text)
        else: #开始了新的一行
            lines.write('\n')            
            if (item_top - top > two_line): # 高度差别比较大，使用两个空行。
                lines.write('\n') 
            lines.write(item_text)           

            start_new_line = True
            top = item_top
            
            delta = (region['rect']['bottom'] - item_top) * 0.7  # 判断是新行的高度
            two_line = delta * 4 # 空两行的高度            
    
    return lines.getvalue()




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


    start_time = time.perf_counter ()

    regions = merge_list(dt_boxes, rec_res_data) # 合并好的区域列表。 [{rect:{left,top,right,bottom}, text:'text'}]
    lines = get_lines(regions) # 合并好的文本
   

    dump = json.dumps({
        # 'image': img,
        'result': {
            #"dt_boxes": dt_boxes,
            #"rec_res_data": rec_res_data,
            "regions": regions,
            "lines": lines
        },
        'info': {
            'total_elapse': elapse,
            'elapse_part': elapse_part,
        },
    })
    
    end_time = time.perf_counter ()
    print('=====耗时',(end_time - start_time)*1000)

    return dump
