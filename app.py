import tensorflow
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from numpy.linalg import norm

import os
from  tqdm import  tqdm
import pickle
model=ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(model.summary())


def extract_feature(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))

    #Convert image into array
    img_array=image.img_to_array(img)

    expanded_img_array=np.expand_dims(img_array,axis=0)

    #give expanded_img_array to preprocessed input
    preprocessed_img=preprocess_input(expanded_img_array)

    # prediction
    result=model.predict(preprocessed_img).flatten()

    #normalize
    normalized_result=result/norm(result)

    return  normalized_result

filename=[]

for file in os.listdir('images'):
    filename.append(os.path.join('images',file))

# print(len(filename))
# print(filename[:5]) #per file path

feature_list  =[]
for file in tqdm(filename):
    feature_list.append(extract_feature(file,model))

# pickle.dump(feature_list,open('embedding.pkl','wb'))
# pickle.dump(filename,open('filename.pkl','wb'))

print(feature_list)
