import dlib
import numpy as np
import cv2
import os

detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('./data enhance/shape_predictor_68_face_landmarks.dat')

# 用来存储生成的单张人脸的路径
path_save = "./data/ALL/"

# def run(img, label, name):

#     face = detector(img, 1)[0]

#     # 计算矩形框大小
#     height = face.bottom() - face.top()
#     width = face.right() - face.left()
#     # height = 256
#     # width = 256

#     # 根据人脸大小生成空的图像
#     img_blank = np.zeros((height, width, 3), np.uint8)

#     for i in range(height):
#         for j in range(width):
#             img_blank[i][j] = img[face.top() + i][face.left() + j]


#     # 存在本地
#     print("Save into:", path_save + label)

#     gray = cv2.cvtColor(img_blank,cv2.COLOR_BGR2GRAY)
#     equ = cv2.equalizeHist(gray)

#     resized=  cv2.resize(equ, (256,256))
#     cv2.imwrite(path_save + label + '/' + name, equ)

def wirte(img, label, name):
    print("Save into:", path_save + label)
    cv2.imwrite(path_save + label + '/' + name, img)

# 读取图像的路径
label_path = '.\\data enhance\\Emotion_labels\\Emotion\\'
img_path = ".\\data enhance\\extended-cohn-kanade-images\\cohn-kanade-images\\"
dict = {0:'neutral', 1:'anger', 2:'contempt', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness', 7:'surprise'}

for root, dirs, files in os.walk(label_path):

    for name in files:
        path = os.path.join(root, name)
        fopen = open(path, 'r')
        label = int(eval(fopen.readline().strip()))
        data = path.replace(label_path, img_path)[:-12] + '.png'

        
        #自然表情
        neutral_data = data[:-6] + '01.png'
        img = cv2.imread(neutral_data)
        wirte(img, dict[0], neutral_data.split('\\')[-1])
        # run(img, dict[0], neutral_data.split('\\')[-1]) 

        # #其他表情
        img = cv2.imread(data)
        wirte(img, dict[label], name[:-12] + '.png')
        # run(img, dict[label], name[:-12] + '.png')    