# importing relevant libraries
import os
import cv2
import time
import numpy as np
from PIL import Image
import streamlit as st
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array


# importing the necessary files
faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
model =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

#creating a function that predicts emotions using the files above
def result(img):
    frame=np.array(img)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    return frame



# a function to read the loaded image
@st.cache    
def load_image(image_):
    pic=Image.open(image_)
    return pic


# main function
def main():
    st.title("Face Emotion Detection App :sunglasses: ")
    st.header("Created by - Avilash Srivastava (5th Sep 2021)")

    activities = ["Play with image", "Play with camera"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Play with image":
        try:
            st.subheader("Please upload an image to detect the emotion.")
            st.info('Make sure to upload an image of a face to get positive result.')
            
            image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg'])

            if image_file is not None:

                img = load_image(image_file)


                st.text("Original Image")
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i+1)
                st.image(image_file)

                if st.button("Process"):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.05)
                        progress.progress(i+1)
                    st.write("Predicted image")
                    result_img = result(img)
                    st.image(result_img, use_column_width = False)
                    st.success("Success")
            else:
                st.warning("No image uploaded yet")
        except Exception:
            st.error('Cannot process this file...Please upload a new image.')
    if choice == "Play with camera":
        st.subheader('Real time test with webcam')
        if st.button("Hit me"):
            try:
                st.info("In case if no window pops up please check your taskbar")
                cap = cv2.VideoCapture(0)
                while True:
                    _, frame = cap.read()
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray)

                    for (x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                        roi_gray = gray[y:y+h,x:x+w]
                        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                        if np.sum([roi_gray])!=0:
                            roi = roi_gray.astype('float')/255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi,axis=0)

                            prediction = model.predict(roi)[0]
                            label=emotion_labels[prediction.argmax()]
                            label_position = (x,y-10)
                            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        else:
                            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    cv2.putText(frame,'Press q to exit',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                    cv2.imshow('Emotion Detector',frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
            except Exception:
                st.error('Opps there seems to be an error!')
                st.warning("Make sure you are using built in camera.")



main()