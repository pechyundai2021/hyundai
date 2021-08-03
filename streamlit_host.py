import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/ball1.h5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'OK',
            1: 'NG'}
          

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(resized, channels="RGB")

    img = image.img_to_array(resized)
    img = img/255  ## normalizing
    img = np.expand_dims(img,axis=0)  
    

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:  
        ypred = model.predict(img)
        ypred = ypred.round()
        a=ypred[0][0]
        st.title("{}".format(map_dict [a]))
