'''
This python script contains code to create a stream lit app. This app detects facial emotions end to end.

To run this file "locally", type 
                    
                    streamlit run app.py --server.maxUploadSize=50
                    
Files required to run this app:
1. app.py
2. haarcascade_frontalface_default.xml
3. model.h5
4. demo_images    (optional) if you want to see demo images
'''


# importing relevant libraries
import os
import av
import cv2
import time
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from time import sleep
from aiortc.contrib.media import MediaPlayer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer



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
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    return frame,label



# a function to read the loaded image
@st.cache    
def load_image(image_):
    pic=Image.open(image_)
    return pic


# a class that captures real time webcam feed
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray)
        if faces is ():
            return img

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y-10)
            cv2.putText(img,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)    
        return img




# main function
def main():
    st.title("Face Emotion Detection App :sunglasses: ")
    st.header("Created by - Avilash Srivastava (5th Sep 2021)")

    activities = ["Play with image", "Play with demo images", "Play with video", "Play with camera"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Play with image":
        st.subheader("Please upload an image to detect the emotion.")
        st.info('Make sure to upload an image of a face to get positive result.')
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg'])
        try:
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
                    result_img,label = result(img)
                    st.image(result_img, use_column_width = False)
                    st.success(f"Success - The person in the image looks '{label}'")
            else:
                st.warning("No image uploaded yet")
        except Exception:
            st.error('Cannot process this file...Please upload a new image.')
        st.info("Or choose a demo image from side pannel")
        
    if choice == "Play with demo images" :
        pictures=["angry1.jpg","angry2.jpg","angry3.jpg","disgust1.jpg",
                    "disgust2.jpg","fear1.jpg","ferimg.jpg","happy1.jpg","happy2.jpg",
                    "neutral1.jpg","neutral2.jpg","sad1.jpg","sad2.jpg","surprise2.jpg"]
        st.subheader("Demo images")
        folder = "./demo_images/"
        c = st.selectbox("Select an image",pictures)
        path=folder+c
        raw=rf"{path}"
        pict = load_image(raw)
        st.image(pict)
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)
        res,l=result(pict)
        st.image(res)



    if choice == "Play with video":
        
        st.subheader("Please upload a video to detect the emotion.")
        st.info('Make sure to upload a video of a person face to get positive result.')
        st.warning("The frame rate might get reduced")

        video_file = st.file_uploader("Upload video")
        try:
            if video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(video_file.read())
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                while cap.isOpened():
                    _, frame = cap.read()
                    if not _:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                        stframe.image(frame)
            else:
                st.warning("No video uploaded yet")
        except Exception:
            st.error("Oops there seems to be an error. Please upload another video")

    if choice == "Play with camera":
        st.subheader("Real time face emotion detection")
        try:
            webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
        except Exception:
            st.error("oops there seems to be an error.")


# calling main function
main()