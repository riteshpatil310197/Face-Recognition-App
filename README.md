# Face-Recognition-App
This project contains End to End Face Recognition App along with deployment using streamlit.

I have created this project with a view to differentiate between faces of myself and one of my friends. However, this project can be extended by adding faces of multiple people along with minor change in **app.py**. 

## Running the Code:
First of all, you need to fork or clone this repository. Then, you have to run **Data_Collector.py** to collect images from webcam of your laptop. **Data_Collector.py** is responsible for creation of data via webcam and therefore, you don't need to collect any sort of external data as such. 

After that, you have to run **face_recognizer.py** to train customised LBPH classifier. 

At last, use **streamlit run app.py** command to run the actual app on local host. You are done !!

## Haar Cascade Classifier:

I have used pre-trained haar cascade classifier for identifying face of the person sitting in front of webcam.

Haar Cascade Classifier is cascade based classifier which uses ensemble learning(Adaboost) to detect features. Each cascade is responsible for detecting particular feature in frame or image(eye,nose,mouth etc.). Since it uses boosting algorithm, it detects face only if all features in the face(learnt during training) is found in particular region of image or frame of video.

### Note:

If you face any kind of problems, don't hesitate to contact me !
 