# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: Max
import base64
import json
import cv2
import numpy as np
from quart import Quart, render_template, request

from task import detect_recognize

app = Quart(__name__)

# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024


@app.route('/')
async def index():
    return await render_template('index.html')


@app.route('/ocr', methods=['POST', 'GET'])
async def ocr():
    if request.method == 'POST':
        url_get =  await request.get_json()
        url_get = str(url_get).split(',')[1]
        print(url_get)
        image = base64.b64decode(url_get)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return  detect_recognize(image)
@app.route('/api', methods=['POST'])
async def ocr_noimage():
    if request.method == 'POST':
        url_get = await  request.get_json()
        url_get = str(url_get).split(',')[1]
        image = base64.b64decode(url_get)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resluts = detect_recognize(image)
        resluts = json.loads(resluts)
        del resluts ['image']
        return json.dumps(resluts,ensure_ascii=False)
if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=9003,
            debug=False,
            processes=True)