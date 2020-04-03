# -*- coding:utf-8 -*-
import cv2
import time
import argparse

from inference import inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection from image")
    parser.add_argument('--img-path', type=str, default='examples/test2.jpg', help='path to your image.')
    args = parser.parse_args()
    imgPath = args.img_path
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inference(img, show_result=True, target_shape=(260, 260))

