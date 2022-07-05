# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rapidocr_onnxruntime import TextSystem
# from rapidocr_openvino import TextSystem


def draw_ocr_box_txt(image, boxes, txts, font_path,
                     scores=None, text_score=0.5):
    if not Path(font_path).exists():
        raise FileNotFoundError(f'The {font_path} does not exists! \n'
                                f'Please download the file in the https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY')

    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < text_score:
            continue

        color = (random.randint(0, 255),
                 random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon([box[0][0], box[0][1],
                            box[1][0], box[1][1],
                            box[2][0], box[2][1],
                            box[3][0], box[3][1]],
                           outline=color)

        box_height = math.sqrt((box[0][0] - box[3][0])**2
                               + (box[0][1] - box[3][1])**2)

        box_width = math.sqrt((box[0][0] - box[1][0])**2
                              + (box[0][1] - box[1][1])**2)

        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size,
                                      encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text((box[0][0] + 3, cur_y), c,
                                fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text([box[0][0], box[0][1]], txt,
                            fill=(0, 0, 0), font=font)

    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def visualize(image_path, boxes, rec_res, font_path="fonts/msyh.ttc"):
    image = Image.open(image_path)
    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]

    draw_img = draw_ocr_box_txt(image, boxes,
                                txts, font_path,
                                scores,
                                text_score=0.5)

    draw_img_save = Path("./inference_results/")
    if not draw_img_save.exists():
        draw_img_save.mkdir(parents=True, exist_ok=True)

    image_save = str(draw_img_save / f'infer_{Path(image_path).name}')
    cv2.imwrite(image_save, draw_img[:, :, ::-1])
    print(f'The infer result has saved in {image_save}')


if __name__ == '__main__':
    text_sys = TextSystem('config.yaml')

    image_path = r'test_images/ch_en_num.jpg'
    img = cv2.imread(image_path)
    dt_boxes, rec_res = text_sys(img)
    print(rec_res)

    visualize(image_path, dt_boxes, rec_res,
              font_path='resources/fonts/msyh.ttc')
