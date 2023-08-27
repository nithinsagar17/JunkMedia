import tensorflow as tf
model = tf.keras.models.load_model('C:\\Users\\bhard\\Desktop\\JMCleaner\\Deep_Learning_Project\\Model.h5')
import streamlit as st
st.write("""
         # Junk File Prediction
         """
         )
st.write("This is a Multi-Class image classification web app to predict type of media file")
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    size = (150,150) 
    img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (331, 331))
    img = tf.reshape(img, (-1, 331, 331, 3))
    prediction = model.predict(img/255)
        
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("""
                 #It is a document image!
                 """
                 
                 )
    elif np.argmax(prediction) == 1:
        st.write("""
                 #It is a Junk image!
                 """
                 
                 )
    elif np.argmax(prediction) == 2:
        st.write("""
                 #It is a Meme image!
                 """
                 
                 )
    elif np.argmax(prediction) == 3:
        st.write("""
                 #It is a Personal image!
                 """
                 
                 )
    else:
        st.write("""
                 #It is a Scenic image!
                 """
                 
                 )
    
    st.text("Probability (0: document, 1: junk, 2: meme, 3: personal, 4: scenic")
    st.write(prediction)