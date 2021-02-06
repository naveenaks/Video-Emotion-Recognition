import os
import cv2
import numpy as np
import xlwt 
from keras.models import model_from_json
from keras.preprocessing import image
from datetime import datetime
from xlwt import Workbook 

wb = Workbook() 
  
sheet1 = wb.add_sheet('Sheet 1') 

style = xlwt.easyxf('font: bold 1') 
  
# Specifying column 
sheet1.write(0, 0, 'Emotion', style)

sheet1.write(0, 1, 'Time Frame', style) 

did = input("Doctor ID :")

pid = input("Patient ID :")

CHECK_FOLDER = os.path.isdir(did)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.mkdir(did)

filename = did+'/'+pid+' video.avi'
frames_per_second = 24.0
res = '720p'

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


model = model_from_json(open("fer2.json", "r").read())

model.load_weights('fer2.h5')

label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


out = cv2.VideoWriter(filename, get_video_type(filename),4.2, get_dims(cap, res))

i = 0

start_time = datetime.now().replace(microsecond=0)

while True:
 
 _,cap_image = cap.read()

 cap_img_gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)

 faces = face_haar_cascade.detectMultiScale(cap_img_gray, 1.3, 5)

 for (x,y,w,h) in faces:

   cv2.rectangle(cap_image, (x,y), (x+w,y+h),(255,0,0),2)
   roi_gray = cap_img_gray[y:y+h, x:x+w]
   roi_gray = cv2.resize(roi_gray, (48,48))
   img_pixels = image.img_to_array(roi_gray)
   img_pixels = np.expand_dims(img_pixels, axis=0)

   predictions = model.predict(img_pixels)
   
   emotion_label = np.argmax(predictions)

   emotion_prediction = label_dict[emotion_label]
   
   cv2.putText(cap_image, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1 )

   current_time = datetime.now().replace(microsecond=0)

   i = i+1
        
   time = str(current_time-start_time)
        
   sheet1.write(i,0,emotion_prediction)
   
   sheet1.write(i,1,time)


   resize_image = cv2.resize(cap_image, (1000,700))
   cv2.imshow('Emotion',resize_image)
   
   out.write(cap_image);
   
 if cv2.waitKey(30) & 0xFF == ord('q'):
    break
  
wb.save(did+"/"+pid+" time.xls")
cap.release()
out.release()
cv2.destroyAllWindows
