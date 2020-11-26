import numpy as np
import os
import cv2


def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('C:/Users/admin/Anaconda3/pkgs/libopencv-3.4.1-h875b8b8_3/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5) #will return the cordinates of the rectangle
    
    return faces,gray_img
def labels_for_training_data(directory):#creating two labels for training data
    faces=[] # all the faces are stored in this 
    faceId=[]
    
    for path,subdirnmes,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue
            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img_path",img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)
            if test_img is None :
             print("Image not loaded properly")
             continue
            faces_rect,gray_img=faceDetection(test_img)
            if (len(faces_rect)!=1):
             continue #since we are assuming only single persoAn image is required
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]# region of interest
            faces.append(roi_gray) # it will accept only a part of face
            faceId.append(int(id))
           
    return faces,faceId       
  
def train_classifier(faces,faceId): #train our classifier using the LBHP recognizer
    face_recognition=cv2.face.LBPHFaceRecognizer_create()
    face_recognition.train(faces,np.array(faceId))
    return face_recognition

def draw_rect(test_img,faces):# draw a rectangle
    (x,y,w,h)=faces
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
 
def put_text(test_img,text,x,y):# used to write text on the images
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,5,(255,0,0),5)
    
    
         