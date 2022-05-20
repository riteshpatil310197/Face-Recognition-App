# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:23:35 2022

@author: Ritesh
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("LBPH.yml")

def detect_faces(input_image):
    conv_img=np.array(input_image.convert("RGB"))
    gray_image=cv2.cvtColor(conv_img,cv2.COLOR_BGR2GRAY)
    
    #Face Detection
    faces=face_detector.detectMultiScale(gray_image)
    name="Bumfuzzle"
    for (x,y,w,h) in faces:
        cv2.rectangle(conv_img,(x,y),(x+w,y+h),(255,0,0), 2)
        id,uncertainty=face_recognizer.predict(gray_image[y:y+h,x:x+w])
        
        if(uncertainty<50):
            if(id==1):
                name="Tanmay"
                cv2.putText(conv_img, name, (x,y+h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,255,0))
            elif(id==2):
                name="Swapnil"
                cv2.putText(conv_img, name, (x,y+h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,255,0))
        else:
                cv2.putText(conv_img, name, (x,y+h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,255,0))
    return conv_img,faces,name

def main():
    st.title("Face Recognition App")
    html_bg = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_bg,unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
    if file is not None:
        image=Image.open(file)
        st.text("Uploaded Image")
        st.image(image)
        
    if st.button("Guess who?"):
        result,face,name=detect_faces(image)
        st.image(result)
        
        
if __name__ == '__main__':
    main()
    

                
        
    
        