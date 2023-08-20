import streamlit as st
import os

from PIL import Image
import pickle
import numpy as np

import tensorflow
from tensorflow import keras
from keras.layers import *
from keras.preprocessing import image
from numpy.linalg import norm
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from sklearn.neighbors import NearestNeighbors



#Model
feature_list=np.array(pickle.load(open('embedding.pkl','rb')))
filename=pickle.load(open('filename.pkl','rb'))

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))

model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('upload',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return  1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert image into array
    img_array = image.img_to_array(img)

    expanded_img_array = np.expand_dims(img_array, axis=0)

    # give expanded_img_array to preprocessed input
    preprocessed_img = preprocess_input(expanded_img_array)

    # prediction
    result = model.predict(preprocessed_img).flatten()

    # normalize
    normalized_result = result / norm(result)

    return normalized_result

def recommend(feature,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([feature])
    return indices



#steps
# File upload->
uploaded_file=st.file_uploader('Choose an image')
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        #display the file
        display_image=Image.open(uploaded_file)
        st.image(display_image)
        #feaature extract
        feature=feature_extraction(os.path.join("upload",uploaded_file.name),model)
        # st.text(feature)
        # recommendation
        indices=recommend(feature,feature_list)
        # show
        #display
        col1,col2,col3,col4,col5=st.columns(5)

        with col1:
            st.image(filename[indices[0][0]])
        with col2:
            st.image(filename[indices[0][1]])
        with col3:
            st.image(filename[indices[0][2]])
        with col4:
            st.image(filename[indices[0][3]])
        with col5:
            st.image(filename[indices[0][4]])

    else:
        st.header("Some error occured in file upload")


