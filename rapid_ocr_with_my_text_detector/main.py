# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: Max
import base64

import time
import cv2
import numpy as np
from flask import Flask, render_template
from flask import g, make_response, request
#from pyinstrument import Profiler
from multiprocessing import Pool

from gevent import monkey
from gevent.pywsgi import WSGIServer # pip install gevent
monkey.patch_all()

from multiprocessing import cpu_count, Process
import datetime
import os


from task import detect_recognize

app = Flask(__name__)

# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024


# @app.before_request
# def before_request():
#     if "profile" in request.args:
#         g.profiler = Profiler()
#         g.profiler.start()


# @app.after_request
# def after_request(response):
#     if not hasattr(g, "profiler"):
#         return response
#     g.profiler.stop()
#     output_html = g.profiler.output_html()
#     return make_response(output_html)


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


@app.route('/ocr', methods=['POST'])
def ocr():
    print("start ocr.")
    
    json_content = request.get_json()
    file_base64 = json_content["file"]
    result = do_ocr(file_base64)

    # with Pool(1) as p:
    #     result = p.map(
    #         do_ocr,
    #         [file_base64]
    #     )

    return result


port = 9003



def run(MULTI_PROCESS):
    if MULTI_PROCESS == False:
        WSGIServer(('0.0.0.0', port), app).serve_forever()
    else:
        mulserver = WSGIServer(('0.0.0.0', port), app)
        mulserver.start()

        def server_forever():
            mulserver.start_accepting()
            mulserver._stop_event.wait()

        for i in range(cpu_count()):
            p = Process(target=server_forever)
            p.start()


if __name__ == "__main__":
    # 单进程 + 协程
    # run(False)
    # 多进程 + 协程
    run(True)

# if __name__ == '__main__':
#     debug = False
#     if debug:
#         app.run(
#             host='127.0.0.1',
#             port=8080,
#             debug=True,
#             threaded=True,
#             # processes=True,
#         )
#     else:
#         host = '0.0.0.0'
#         port = 9003
#         print(
#             f'launching server... on {host}:{port}'
#         )
        

#         from waitress import serve # pip install waitress
#         serve(
#             app,
#             host=host,
#             port=port,
#             threads=20
#         )