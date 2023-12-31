import pickle
import numpy as np

import tensorflow
from tensorflow import keras
from keras.layers import *
from keras.preprocessing import image
from numpy.linalg import norm
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from sklearn.neighbors import NearestNeighbors

import cv2

feature_list=np.array(pickle.load(open('embedding.pkl','rb')))
filename=pickle.load(open('filename.pkl','rb'))

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))

model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img=image.load_img('sample/1163.jpg',target_size=(224,224))
img_array=image.img_to_array(img)
expanded_img_array=np.expand_dims(img_array,axis=0)
preprocessed_img=preprocess_input(expanded_img_array)
result=model.predict(preprocessed_img).flatten()
normalized_result=result/norm(result)


neighbors=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')

neighbors.fit(feature_list)


distances,indices=neighbors.kneighbors([normalized_result])

print(indices)
# print(distances)

for file in indices[0]:
    print(filename[file])
    temp_img = cv2.imread(filename[file])
    resized_img=cv2.resize(temp_img,(512,512))
    cv2.imshow('output',resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

