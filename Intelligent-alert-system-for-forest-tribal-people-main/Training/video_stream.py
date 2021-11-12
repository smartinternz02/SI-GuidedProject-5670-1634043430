#import opencv
import os
import cv2
#import numpy
import numpy as np
from keras.preprocessing import image 
from tensorflow.keras import Sequential
from keras.models  import load_model
#import Client from twilio API
from twilio.rest import Client, TwilioHttpClient
#Load saved model file using load_model method
model = load_model('alert.h5')
#To read webcam
video = cv2.VideoCapture(0)
#Type of classes or names of the labels that we considered
name = ['Human','Domestic', 'Wild']
#To execute the program repeatedly using while loop   
while(1):
    success, frame = video.read()
    cv2.imwrite("image.jpg",frame)
    img = image.load_img("image.jpg",target_size = (64,64))
    x  = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    pred = np.argmax(model.predict(x),axis=-1)
    p = pred[0]
    print(pred)
    cv2.putText(frame, "predicted  class = "+str(name[p]), (100,100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
    
    pred = np.argmax(model.predict(x),axis=-1)
    if pred[0]==2:
        #twilio account ssid
         
        account_sid = 'ACac0b11bf8d5fa7bcb60e8c6731c5fc84'
        #twilo account authentication toke
        auth_token = '80e6e545010cdbae0b421a977f0145b8'
        client =Client(account_sid, auth_token)

        message = client.messages \
        .create(
         body='Wild animal detected',
         from_='+13203993780', #the free number of twilio
         to='+916302625267')
        print(message.sid)
        print('Danger!!')
        print('Animal Detected')
        print ('SMS sent!')
        #break
    else:
        print("No Danger")
       #break
    cv2.imshow("image",frame)
    if cv2.waitKey(1) & 0xFF == ord('a'): 
        break

video.release()
cv2.destroyAllWindows()