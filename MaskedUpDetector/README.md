# Face Mask Detector using MobileNet v2 and DNNs
This project is a face mask detector that uses machine learning to detect whether a person is wearing a face mask or not. The detector is built using two different models, a Deep Neural Network (DNN) model published in OpenCV's GitHub that can detect faces, and a basic Convolutional Neural Network (CNN) and MobileNet v2, which are trained on a dataset of masked and unmasked people. Thereby, the project implements both transfer learning and a multi model solution. This project is also available as a [Kaggle Notebook](https://www.kaggle.com/code/shibinjudah/face-mask-detector-using-mobilenet-v2).

## Dataset
The dataset used in this project is obtained from [Kaggle's Face Mask Detection](https://www.kaggle.com/datasets/21faa9e463f87c2500de415965f97074cc83502d0f10766fb62a2e1c2bc6b512) dataset. It consists of images of people with and without face masks. The dataset is divided into training and testing sets in a 80:20 split.

## Models
1. **Basic CNN:** 
The basic CNN model is built using the Keras library in Python. It uses a Data Augmentation layer to improve variance in the dataset, along with Average 2D Pooling, Dropout and Dense layers.  

2. **MobileNet v2:** 
The MobileNet v2 model is a pre-trained model that uses transfer learning to improve accuracy. It is also built using Keras and uses the 'ImageNet' weights and is used along the basic CNN without any training. MobileNet v2 is used for this application as its performance as a mobile model is state of the art. 

3. **OpenCV DNN Model and Weights:** 
The OpenCV DNN model is published in OpenCV's [GitHub](https://github.com/opencv) repository. Its a pre-trained model and is available as a [Prototxt file](MaskedUpDetector/model/deploy.prototxt) and its weights published as a [caffemodel file](MaskedUpDetector/model/res10_300x300_ssd_iter_140000.caffemodel) to detect faces in images.

Using transfer learning, the Basic CNN model and the MobileNet v2 models are used together to train on the dataset and is saved as the [Mask detector](MaskedUpDetector/model/mask_detector.model)

## Dependencies
To run this project, you will need the following dependencies:

- Python 3.x
- Numpy
- OpenCV
- TensorFlow
- Keras
- Scikit-learn
- Matplotlib

You can install these dependencies using pip by running the following command:
```
pip install numpy opencv-python tensorflow keras scikit-learn matplotlib
```
## Usage
To train the model, run the training.py script. The trained model will be saved in the model directory.
```
python training.py
```
To use the face mask detector, simply run the Python script provided in the project's folder. The script loads the trained models and uses them to predict whether a person in the webcam is wearing a mask or not.
```
python main.py
```

## Results
The model acheived 97% and 98% accuracy on training and test data respectively, with very minimal loss. This shows the power of transfer learning, where a simple model along with a general model can leverage better accuracy in just a few epochs of training. The plots and the classification reports are available in the [results folder](/result)

## Future Work
- Leverage the 'Mask-not-worn-properly' dataset to identify improperly worn masks.
- Create a smart security cam app for better and wider usability.  

## Conclusion
This face mask detector project demonstrates the use of machine learning in solving real-world problems. By building and training two different models and using a pre-trained DNN model, we can accurately detect whether a person is wearing a face mask or not. This project can serve as a valuable resource for anyone interested in machine learning and computer vision applications.
