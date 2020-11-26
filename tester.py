
import cv2
import faceRecognition as fr

test_img=cv2.imread('C:/Users/admin/Desktop/python project/TestImages/4.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("face_detected:",faces_detected)

for(x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
    
'''resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("facedetectiontuorial",resized_img)
cv2.waitKey(0);
cv2.destroyAllWindows()'''



faces,faceId=fr.labels_for_training_data("C:/Users/admin/Desktop/python project/Trainingimages")
face_recognizer=fr.train_classifier(faces,faceId)
#face_recognizer.save('trainingData.yml')
 #calling train classifier function
face_recognize=cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('C:/Users/admin/Desktop/python project/Face Recognition/trainingData.yml')
name={0:"Ali Husain",1:"Shahrukh Khan"}
 
for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    
    label,confidence=face_recognizer.predict(roi_gray) #if confidence value is zero then exact match    
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>90):
     continue
    fr.put_text(test_img,predicted_name,x,y)

for(x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(247, 0, 0),thickness=1)
    
resized_img=cv2.resize(test_img,(1000,800))
cv2.imshow("facedetectiontuorial",resized_img)
cv2.waitKey(0);
cv2.destroyAllWindows()
 

