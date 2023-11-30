# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:07:09 2020

@author: Shibin Judah Paul
"""


# import tensorflow
# import keras
# import imutils
# import numpy as np
# import cv2
# import matplotlib
# import scipy
# import os


# print("tensorflow", tensorflow.__version__)
# print("keras", keras.__version__)
# print("imutils", imutils.__version__)
# print("np", np.__version__)
# print("cv2", cv2.__version__)
# print("matplotlib", matplotlib.__version__)
# print("scipy", scipy.__version__)


# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm 
import time 



    
# Initialize the initial learning rate - 
# Learning rate, generally represented by the symbol ‘α’, is a hyper-parameter
# used to control the rate at which an algorithm updates the parameter estimates 
# or learns the values of the parameters.   

# number of epochs to train for -
# 1 epoch is an entire length of a dataset.

# and batch size
# Total number of training examples present in a single batch.

    
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


DIRECTORY = "../dataset/"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print(">>> loading images")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY,category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(244,244))
        image = img_to_array(image)
        image = preprocess_input(image) #PP req for mobile_net
        
        data.append(image)
        labels.append(category)
        
        
#Binarize labels in a one-vs-all fashion
Label_Bi = LabelBinarizer()
labels = Label_Bi.fit_transform(labels)
labels = to_categorical(labels)

#numpy conversion
data = np.array(data, dtype="float32")
labels = np.array(labels)

# #Split arrays or matrices into random train and test subsets
# stratify:array-like, default=None
#          If not None, data is split in a stratified fashion, 
#           using this as the class labels.

#random_state:int or RandomState instance, default=None
#       Controls the shuffling applied to the data before applying the split. 
#       Pass an int for reproducible output across multiple function calls. 

# test_size:float or int, default=None
# If float, should be between 0.0 and 1.0 and represent the proportion 
# of the dataset to include in the test split. 
#If int, represents the absolute number of test samples. 
#If None, the value is set to the complement of the train size. 
#If train_size is also None, it will be set to 0.25.

(X_train, X_test, Y_train, Y_test) = train_test_split(data,
    labels,test_size=0.20, stratify = labels, random_state = 42)


# Data Augmentation is a technique of creating new data from existing data 
# by applying some transformations such as flips, rotate at a various angle,
# shifts, zooms and many more. Training the neural network on more data leads
# to achieving higher accuracy. In real-world problem, we may have limited data.
# Therefore, data augmentation is often used to increase train dataset.

AugmentedData = ImageDataGenerator(
            	rotation_range=20,
            	zoom_range=0.15,
            	width_shift_range=0.2,
            	height_shift_range=0.2,
            	shear_range=0.15,
            	horizontal_flip=True,
            	fill_mode="nearest")

# Using 2 models Base and Head models:
    
# MobileNet-v2 is a convolutional neural network that is 53 layers deep.
# You can load a pretrained version of the network trained on more than
# a million images from the ImageNet database [1]. 
# The pretrained network can classify images into 1000 object categories,
# As a result, the network has learned rich feature representations for a wide
# range of images. The network has an image input size of 224-by-224.


baseModel = MobileNetV2(weights="imagenet", include_top=False, 
                        input_tensor= Input(shape=(224, 224, 3)))

# The head model receives the base's op and is processed.
#
# Pooling is basically “downscaling” the image obtained from the previous layers.
# It can be compared to shrinking an image to reduce its pixel density.
#   -MaxPooling: each block, or “pool”, the operation simply involves 
#        computing the max value.
#   -AvgPooling: each block, or “pool”, the operation simply involves 
#        computing the avg value
#
# AveragePooling2D layer(): Average pooling operation for spatial data.

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)

#Flatten(): returns a copy of the array in one dimensional
# rather than in 2-D or a multi-dimensional array.

headModel = Flatten()(headModel)

#dense(): creates a dense conv net with given neurons and activation fn.

headModel = Dense(128,activation="relu")(headModel)

#Dropout(): Dropout is one of the important concept in the machine learning. 
# It is used to fix the over-fitting issue. Input data may have some of the 
# unwanted data, usually called as Noise. Dropout will try to remove the noise 
# data and thus prevent the model from over-fitting.

headModel = Dropout(0.5)(headModel)

#Using dense() to create a less denser conv net for the output layer.
# softmax and sigmoid fns are best suited for binary classifications.

headModel = Dense(2, activation="softmax")(headModel)

# Fuse both models

fusedModel = Model(inputs= baseModel.input, outputs= headModel)

#Since baseModel (imagenet) is a pretrained model, we skip its training in
# the first iteration by freezing the model'

for layer in baseModel.layers:
    layer.trainable = False


#compile model
print(">>> Compiling Model")

#Adam Optimizer:  
# Adaptive Moment Estimation is an algorithm for optimization technique
# for gradient descent. The method is really efficient when working with 
# large problem involving a lot of data or parameters. It requires less 
# memory and is efficient. Intuitively, it is a combination of the 
# ‘gradient descent with momentum’ algorithm and the ‘RMSP’ algorithm.

optimized = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

#Loss function is used to find error or deviation in the learning process.
# Keras requires loss function during model compilation process.

#Optimization is an important process which optimize the input weights 
# by comparing the prediction and the loss function.
# Keras provides quite a few optimizer as a module

#Metrics is used to evaluate the performance of your model. 
# It is similar to loss function, but not used in training process. 
# Keras provides quite a few metrics as a module


fusedModel.compile(loss="binary_crossentropy", optimizer=optimized,
	metrics=["accuracy"])

#Training headModel
print(">>> Training headModel")

#Model.fit() gets the model to fit into the given data
H = fusedModel.fit(
	AugmentedData.flow(X_train, Y_train, batch_size=BS),
	steps_per_epoch=len(X_train) // BS,
	validation_data=(X_test, Y_test),
	validation_steps=len(X_test) // BS,
	epochs=EPOCHS)


#Prediction is the final step and our expected outcome of the model generation.
# Keras provides a method, predict to get the prediction of the trained model.

print(">>> Generating Test predictions")
predicted_Index = fusedModel.predict(X_test, batch_size=BS)

#The numpy.argmax() function returns indices of the max element of the array 
# in a particular axis.
#Return: Array of indices into the array with same shape as array.shape()
# with the dimension along axis removed.

#In context, for each image evaluated, we obtain the index of the highly 
# probable label for that image.

predicted_Index = np.argmax(predicted_Index, axis=1)

#Build a text report showing the main classification metrics.
print(classification_report(Y_test.argmax(axis = 1), predicted_Index, 
                            target_names= Label_Bi.classes_))

#Model.Save(): It saves data of the model ina serialized manner.
fusedModel.save("../model/mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
