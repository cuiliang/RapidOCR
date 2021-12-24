# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: Max
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request

from task import detect_recognize

app = Flask(__name__)

# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024


@app.route('/')
def index():
    return "Hello world."


@app.route('/ocr', methods=['POST'])
def ocr():
    if request.method == 'POST':
        url_get = request.get_json()
        url_get = str(url_get).split(',')[1]

        image = base64.b64decode(url_get)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return detect_recognize(image)


if __name__ == '__main__':
    debug = False
    if debug:
        app.run(host='127.0.0.1',
                port=9003,
                debug=True,
                processes=True)
    else:
        host = '0.0.0.0'
        port = 8080
        print(
            f'launching gevent server... on {host}:{port}'
        )
        
        from gevent.pywsgi import WSGIServer # pip install gevent
        http_server = WSGIServer((host, port), app)
        http_server.serve_forever()
