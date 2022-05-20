import cv2

stream=cv2.VideoCapture(0,cv2.CAP_DSHOW)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ID= input("Enter your Number:")
count=0
while(True):
    ret,img=stream.read()
    gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray_image)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

        #Save image
        cv2.imwrite("dataset/User."+ID +'.'+ str(count) + ".jpg", gray_image[y:y+h,x:x+w])
        count+=1
        cv2.imshow('frame',img)
    if count>60:
        break
stream.release()
cv2.destroyAllWindows()