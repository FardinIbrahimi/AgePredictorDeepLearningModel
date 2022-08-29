
from decimal import ROUND_05UP, ROUND_UP, Rounded
from distutils.command.upload import upload
from email.mime import image
from msilib.schema import File
from unicodedata import decimal
import streamlit as st
import pandas as pd
#from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from keras.models import load_model
import numpy as np
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

st.set_page_config(page_title='Age Predictor App',layout='wide')
st.write("""
# Age Predictor App""")    

data = pd.read_csv('age_gender.csv')
X = data[['pixels']]
y = data['age']


st.sidebar.write("Upload an image and predict his/her age!")

    
    #st.markdown('The Diabetes dataset used for training the model is:')
    #st.write(data.head(5))
    
st.markdown('**1.Dataset - We have used only the age column from the dataset!**')
st.write(data.head(10))
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
     #st.sidebar.header('2. Set Parameters'):
age = st.sidebar.file_uploader("Upload An Image!", type = ['jpg', 'png'])
if age is not None:
    file_bytes = np.asarray(bytearray(age.read()), dtype = np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(opencv_image, (48,48))
    st.sidebar.image(age)
    resized_image = mobilenet_v2_preprocess_input(resized_image)
    image_reshaped = resized_image[np.newaxis, ...]
    Prediction_button = st.sidebar.button ("Predict Age")
    model = load_model("age_model.h5")
    if Prediction_button:
        prediction = model.predict(image_reshaped)
        prediction = np.round_(prediction)
        st.sidebar.title(f"{prediction} Years Old")
        if prediction < 10:
            st.sidebar.write("Child")
        elif prediction >= 10 and prediction < 18:
            st.sidebar.write("Teenager")
        elif prediction >= 18 and prediction < 40:
            st.sidebar.write("Young")
        elif prediction >= 40 and prediction < 60:
            st.sidebar.write("Middle Age")
        elif prediction >= 60 and prediction < 75:
            st.sidebar.write("Old")
            
        else:
            print ("Very Old")



    #logregs = LogisticRegression()
    #logregs.fit(X_train, y_train)
    #y_pred_st = logregs.predict(X_test_sc)
    
        st.sidebar.title("Created By:")
        st.sidebar.subheader("Fardin Ibrahimi")
