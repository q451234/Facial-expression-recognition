#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:
    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
See the above for an explanation of the code below.
To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:
    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:
    ./faceswap.py <head image> <face image>
If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.
"""

import os
import random
import cv2
import dlib
import numpy
from shutil import copy,move
from PIL import Image

PREDICTOR_PATH = "./data enhance/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11



LEFT_EYE_POINTS = list(range(42, 48)) + list(range(22, 31))
RIGHT_EYE_POINTS = list(range(36, 42)) + list(range(17, 21)) + list(range(27, 31))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61)) + list(range(6, 10)) + list(range(31, 35))
EYES_NOSE_POINTS = list(range(17, 35))
NOSE_MOUTH_POINTS = list(range(28, 35)) + list(range(3, 14))
FACE_POINTS = list(range(0, 68))

dict = {"left_eye" : LEFT_EYE_POINTS, "right_eye" : RIGHT_EYE_POINTS, 
        "nose" : NOSE_POINTS, "mouth" : MOUTH_POINTS, 
        "eyes_nose" : EYES_NOSE_POINTS, "nose_mouth" : NOSE_MOUTH_POINTS, 
        "face" : FACE_POINTS}

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [LEFT_EYE_POINTS, RIGHT_EYE_POINTS, NOSE_POINTS, MOUTH_POINTS, EYES_NOSE_POINTS, NOSE_MOUTH_POINTS, FACE_POINTS]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.boundingRect(points)
    cv2.rectangle(im, points, 1, thickness=-1)

def get_face_mask(im, landmarks, roi):
    im = numpy.zeros(im.shape[:2], dtype=numpy.uint8)

    draw_convex_hull(im, landmarks[roi], color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))
    return im

def read_im_and_landmarks(fname):
    im = cv2.imread(fname)
    s = get_landmarks(im)

    return im, s

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

def generate_roi(img, roi):
    img, landmarks = read_im_and_landmarks(img)

    mask = get_face_mask(img, landmarks, roi)
    res = img * mask

    res = cv2.resize(res, (256,256))
    return res

file_path = 'data/ALL'
expression_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]

mkfile('data/train')
for cla in expression_class:
    for roi in dict.keys():
        mkfile('data/train/' + roi + '/' + cla)
    
# 创建 验证集val 文件夹，并由5种类名在其目录下创建5个子目录
mkfile('data/val')
for cla in expression_class:
    for roi in dict.keys():
        mkfile('data/val/' + roi + '/' + cla)

mkfile('data/test')
for cla in expression_class:
    for roi in dict.keys():
        mkfile('data/test/' + roi + '/' + cla)

split_rate = 0.2

for cla in expression_class:
    cla_path = file_path + '/' + cla + '/'  # 某一个表情的子目录
    images = os.listdir(cla_path)		    # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate)) # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
    	# eval_index 中保存验证集val的图像名称
        if image in eval_index:					
            image_path = cla_path + image
            for roi in dict.keys():
                roi_img = generate_roi(image_path, dict[roi])
                new_path = 'data/val/' + roi + '/' + cla + '/' + image                
                cv2.imwrite(new_path, roi_img)
           
        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            for roi in dict.keys():
                roi_img = generate_roi(image_path, dict[roi])
                new_path = 'data/train/' + roi + '/' + cla + '/' + image                
                cv2.imwrite(new_path, roi_img)           

        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()

val_path = 'data/val'

for cla in expression_class:
    cla_path = val_path + '/face/' + cla +'/' # 某一个表情的子目录
    images = os.listdir(cla_path)		    # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    eval_index = random.sample(images, k=int(num*0.5)) # 从images列表中随机抽取 k 个图像名称
    # print(eval_index)
    for index, image in enumerate(images):
        for roi in dict.keys():
            if image in eval_index:					
                image_path = cla_path.replace('face', roi) + image
                new_path = 'data/test/' + roi + '/' + cla + '/' + image
                move(image_path, new_path)  # 将选中的图像复制到新路径

            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
print("processing done!")
