from __future__ import print_function, division
import os
from os import listdir
from sklearn import metrics
import pylab
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPool2D, BatchNormalization, Dropout
from keras.utils import  to_categorical

import warnings
warnings.filterwarnings("ignore")

def imageFeature(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predict = model.predict(x)
    return predict


path = '/Users/andrey/And/ImageBase/train/'

classes = os.listdir((path))

labels = {}
for d in range(len(classes)):
    labels[classes[d]] = d

res_table = []
X = []
Y = []
for d in classes:
    if d == '.DS_Store':
        continue
    for image1 in os.listdir(path+d):
        if image1.split('.')[1] != 'jpg':
            continue
        img = image.load_img(path+d + '/' + image1, target_size=(64, 64, 3))
        X.append(image.img_to_array(img))
        Y.append(labels[d])

Y = to_categorical(Y)
# print(res_table)
X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(13, activation='softmax'))
model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X, Y, batch_size=32, epochs=10)

img = image.load_img("/Users/andrey/And/ImageBase/train/Gladiolus/Gladiolus_0001.jpg", target_size=(64, 64, 3))
img = image.img_to_array(img)
print(model.predict(img.reshape(1, 64, 64, 3)))

# imageFeature('/Users/andrey/And/ImageBase/train/Gladiolus/Gladiolus_0001.jpg')



# def queryFeature(path):
#     # img = image.load_img(img_path, target_size=(224, 224))
#     for d in classes:
#         if d == '.DS_Store':
#             continue
#         for image1 in os.listdir(path + d):
#             if image1.split('.')[1] != 'jpg':
#                 continue
#             img = image.load_img(path + d + '/' + image1)
#
#             img = image.img_to_array(img)
#             img = np.expand_dims(img, axis=0)
#             img = preprocess_input(img)
#             features = model.predict(img)
#     return features


# path = '/Users/andrey/datasets/oxford/image/oxbuild_images/all_souls_000013.jpg'


# mainVector = queryFeature(path)
# featureList = imageFeature(path)

# for i in range(len(featureList)):
#
#   # cosMetric = metrics.pairwise.cosine_similarity(mainVector.reshape(1, -1), featureList[i].reshape(1, -1))
#    cosMetric = metrics.pairwise.cosine_similarity(mainVector, featureList[i])
#    print(cosMetric)

