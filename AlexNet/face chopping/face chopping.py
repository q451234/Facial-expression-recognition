import os
import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

cascPath = "./face chopping/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
predictor_landmarks = dlib.shape_predictor("./data enhance/shape_predictor_68_face_landmarks.dat")

def pre_processing(img):
    cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
    # img = cv2.equalizeHist(img)
    return img


def chopping(img):
    read_img = cv2.imread(img)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        read_img,
        scaleFactor=1.1,
        minNeighbors=6,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if(len(faces) != 0):
        (x, y, w, h) = faces[0]  
        result = read_img[y:y+h, x:x+w]
        img_path = img.replace('CK+','data/ALL')
        print(img_path)

        result = cv2.resize(result, (256,256))
        result = pre_processing(result)
        
        cv2.imwrite(img_path, result)
    else:
        print("No Face")

dict = {0:'neutral', 1:'anger', 2:'contempt', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness', 7:'surprise'}
data_path = "./CK+/"

for root, dirs, files in os.walk(data_path):
    for name in files:
        path = os.path.join(root, name)
        chopping(path)      
