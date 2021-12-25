# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: Max
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

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


@app.route('/')
def index():
    return "Hello world."

@app.route("/cppla", methods=['GET'])
def function_benchmark():
    return jsonify(
        {
            "status": "ok",
            "time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            "pid": os.getpid()
        }
    ), 200


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

@app.route('/ocr1', methods=['POST'])
def ocr1():
    if request.method == 'POST':
        url_get = request.get_json()
        url_get = str(url_get).split(',')[1]

        image = base64.b64decode(url_get)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return detect_recognize(image)

port = 9003


from gevent.socket import socket
listener = socket()

def serve_forever(listener):
    WSGIServer(listener, app).serve_forever()

number_of_processes = 5

for i in range(number_of_processes):
    Process(target=serve_forever, args=(listener,)).start()

serve_forever(listener)




# def run(MULTI_PROCESS):
#     if MULTI_PROCESS == False:
#         WSGIServer(('0.0.0.0', port), app).serve_forever()
#     else:
#         mulserver = WSGIServer(('0.0.0.0', port), app)
#         mulserver.start()

#         def server_forever():
#             mulserver.start_accepting()
#             mulserver._stop_event.wait()

#         for i in range(cpu_count()):
#             p = Process(target=server_forever)
#             p.start()


# if __name__ == "__main__":
#     # 单进程 + 协程
#     # run(False)
#     # 多进程 + 协程
#     run(False)


'''
##################################################

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
        
        
        http_server = WSGIServer((host, port), app)
        http_server.serve_forever()
'''