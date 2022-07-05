# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request

from api_task import detect_recognize

from multiprocessing import Pool
from multiprocessing import cpu_count, Process

from gevent import monkey
from gevent.pywsgi import WSGIServer # pip install gevent
monkey.patch_all()



app = Flask(__name__)

# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# 服务端口号
port = 9003

@app.route('/')
def index():
    #return render_template('index.html')
    print("index")
    return "Hello world from RapidOCR server. "

@app.route('/ocr', methods=['POST', 'GET'])
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


#####################################################



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

#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=9003, debug=False, processes=True)
