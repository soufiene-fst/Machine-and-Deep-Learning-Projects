# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:55:44 2020

@author: Shibin Judah Paul
"""
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


# load a pre-existing serialized face detector model from disk and the weights
faceDetector = "..\model\deploy.prototxt"
faceDetector_weights = "..\model\res10_300x300_ssd_iter_140000.caffemodel"

#Read the Model and configuration(weights) and returns a Net object
# Model - Binary file contains trained weights.(*.caffemodel)
# config - Text file contains network configuration.(*.prototxt)

faceNet = cv2.dnn.readNet(faceDetector, faceDetector_weights)

# load our own face mask detector model from disk
maskNet = load_model("..\model\mask_detector.model")

def detect_Masks(givenFrame, faceNet, maskNet):
    #obtain image dimensions and create a blob
    (height, width) = givenFrame.shape[:2]
    blob = cv2.dnn.blobFromImage(givenFrame, 1.0, (224,224),
                                 (104.0, 177.0, 123.0))
    
    #Use faceNet to first identify faces to obtain the relevant face data
    #from it. all faces, location of the faces and their corresponding predictions.
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    
    #Initialize empty lists to hold face data
    faces = []
    face_coordinates = []
    predictions = []
    
    #loop over each detection to obtain data one by one.
    # img.shape rets H,W and No. of channels(= no. of detections in an image)
    for i in range(0, detections.shape[2]):
        # extract the probability associated with each detections
        # forward() returns an array [0, 0, Index, 1] : Index holds the no. 
        # of detections, last value: 1- prediction class,
        # 2 - confidence factor, 3-7 has the bounding box coordinates.
        # using "i" to target the cur detection's probability by using "2"
        # in the last value
        confidence = detections[0, 0, i, 2]
        
        
        #filtering out weaker prediction
        if confidence > 0.7:
            boundingBox = detections[0,0,i,3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = boundingBox.astype("int")
            
            # ensure the begining coordinates are not 0 and ending coordinates
            # are atleast 1px less than the actual width(x-axis) and height(y-axis)
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(width - 1, endX), min(height - 1, endY))
            
            
            #Now that we have the X,Y: we have the region of interest.
            #we can strip this part and switch it to required color channels
            # BGR to RGB, blow it up to 224x224, image to np array and preprocess it.
            facialROI = givenFrame[startY:endY, startX:endX]
            facialROI = cv2.cvtColor(facialROI, cv2.COLOR_BGR2RGB)
            facialROI = cv2.resize(facialROI, (224, 224))
            facialROI = img_to_array(facialROI)
            facialROI = preprocess_input(facialROI)
            
            #update respective lists with data obtained
            faces.append(facialROI)
            face_coordinates.append((startX, startY, endX, endY))
        
        if len(faces) > 0:
            #Creating batch predictions for faster run time
            faces = np.array(faces, "float32")
            predictions = maskNet.predict(faces, batch_size=32)
        
        return(face_coordinates, predictions)

# initialize the video stream
print(">>> starting video stream...")
vidStream = VideoStream(src=0)
vidStream.start()

while True:
    #Get frames from the stream and resize them to 400px
    currFrame = vidStream.read()
    currFrame = imutils.resize(currFrame, width= 400)
    
    #Use the method to obtain coordinates and prediction values
    (face_coordinates, predictions) = detect_Masks(currFrame, faceNet, maskNet)
    
    #looping over all face_coor and respective predictions
    #Zip() indexes a list of values with another list.
    for (boBo, currPred) in zip(face_coordinates, predictions):
        #unpacking the lists into individual vars
        (startX, startY, endX, endY) = boBo
        (w_mask, wo_mask) = currPred
        
        #assign class label based on prediction
        label = "[Mask:Yes]" if w_mask > wo_mask else "[Mask: No]"
        color = (0, 255, 0) if label == "[Mask:Yes]" else (0, 0, 255)
        
        # Adding confidence rate
        label = "{}: {:.2f}%".format(label, max(w_mask, wo_mask) * 100)
        
        # use CV2 methods to draw box, add label, its font, font scale, color 
        # and thickness
        cv2.putText(currFrame, label, (startX, startY - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,2)
        cv2.rectangle(currFrame, (startX, startY), (endX, endY), color,2)
    
    #display output
    cv2.imshow("Face Mask Detector", currFrame)
    key = cv2.waitKey(1) & 0xFF
    
    if key== ord("q"):
        cv2.destroyAllWindows()
        vidStream.stop()
        break

cv2.destroyAllWindows()
vidStream.stop()
