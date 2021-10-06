# Face-Emotion-Recognition
Deep Learning +MLE capstone project [Almabetter pro]

## Introduction:
Here I have built a real time face emotion recognition training model and application to detect emotions of a person using a camera. The model was trained on [FER-2013](https://www.kaggle.com/msambare/fer2013) dataset which had a total of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgust, fear, happy, neutral, sad and surprise. Custome built Convolutional Neural Network architecture was used to train the model.

![image](https://user-images.githubusercontent.com/88347331/136237210-1c6134ed-9bde-4216-8923-ae6dac5b1a4d.png)

The model gave an accuracy of 76% for training set and 67% accuracy for test set. A web application was built and deployed on Azure and Streamlit cloud using streamlit API.

## How to run locally:

To run the script you must have python or [anaconda](https://www.anaconda.com/products/individual) installed. After installation open 'anaconda prompt'

* First, clone the repository and enter the folder

```
git clone https://github.com/avilashsrivastava/Face-Emotion-Recognition.git
cd Face-Emotion-Recognition
```

* Install the dependencies
    `pip install -r requirements.txt`
    
* run the webcam
    `python camera.py`

## Create a docker image:

I have also put a Dockerfile which you can use to build a docker image. To build an image you must download [Docker](https://www.docker.com/products/docker-desktop)

* After installating docker go to terminal or command prompt
* Go inside the cloned folder `cd Face-Emotion-Recognition`
* Type the following to build a docker image
```
docker build -t appname:version
```

# Check out the deployed app:

Azure cloud - https://real-time-face-emotion-recognition-2021.azurewebsites.net

Streamlit cloud - https://share.streamlit.io/avilashsrivastava/face-emotion-recognition/main/app.py

Note - It might take few minutes to load when opening first time. After few minutes reload the page and it should be up running.
I only have free account and less compute resources so the app might be slow.
