# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:48:57 2022

@author: Ritesh
"""

import cv2
import os
import numpy as np
from PIL import Image

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataset'

def prepareData(path):
    image_paths=[os.path.join(path, f) for f in os.listdir(path)]
    faces=[]
    IDS=[]
    for single in image_paths:
        img=Image.open(single).convert('L')
        img_arr=np.array(img,'uint8')
        ID=int(os.path.split(single)[-1].split(".")[1])
        faces.append(img_arr)
        IDS.append(ID)
        cv2.imshow("train", img_arr)
        cv2.waitKey(10)
    return np.array(IDS),faces

IDS,faces=prepareData(path)
face_recognizer.train(faces,IDS)
face_recognizer.save('LBPH.yml')
cv2.destroyAllWindows()


        
