# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: Max
import base64

import time
import cv2
import numpy as np
from flask import Flask, render_template, request

from task import detect_recognize

app = Flask(__name__)

# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024


@app.route('/')
def index():
    time.sleep(2)
    print("index")
    return "Hello world."

def do_ocr(img_base64):
    img_base64 = str(img_base64).split(',')[1]

    image = base64.b64decode(img_base64)
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return detect_recognize(image)


# @app.route('/ocr', methods=['POST'])
# def ocr():
#     print("start ocr.")
#     if request.method == 'POST':
#         json_content = request.get_json()
#         file_base64 = json_content["file"]
#         result = do_ocr(file_base64)
#         return result

from multiprocessing import Pool

@app.route('/ocr', methods=['POST'])
def ocr():
    print("start ocr.")
    if request.method == 'POST':
        json_content = request.get_json()
        file_base64 = json_content["file"]
        # result = do_ocr(file_base64)       
        #return result

        with Pool(1) as p:
            result = p.map(
                do_ocr,
                [file_base64]
            )
        return result[0]

if __name__ == '__main__':
    debug = True
    if debug:
        app.run(
            host='0.0.0.0',
            port=9003,
            debug=True,
            threaded=True,
            # processes=True,
        )
    else:
        host = '0.0.0.0'
        port = 8080
        print(
            f'launching server... on {host}:{port}'
        )
        

        from waitress import serve # pip install waitress
        serve(
            app,
            host=host,
            port=port,
            threads=20
        )