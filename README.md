# CIFAR10-Classification
This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to perform multi-class image classification on the CIFAR-10 dataset.
The model is trained to classify images into 10 different object categories.

1.Project Overview:
  Task: Multi-class image classification
  Dataset: CIFAR-10
  Framework: PyTorch
  Model Type: Custom CNN
  Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

2.Model Architecture
The CNN architecture consists of:
  Convolutional Feature Extractor
    -3 Convolution layers: Conv2D → ReLU → MaxPooling
    -Channel progression: 3 → 32 → 64 → 128
    -Spatial size reduced from 32×32 → 4×4
  Fully Connected Classifier
    -Flatten layer  
    -Dense layer (256 units + ReLU)
    -Output layer with 10 neurons (one per class)

3.Dataset Details
CIFAR-10 contains:
  -60,000 color images (32×32)
  -50,000 training images
  -10,000 test images
Dataset is automatically downloaded using torchvision.datasets

4.Data Preprocessing:
The following transformations are applied:
  -Resize images to 32×32
  -Convert images to tensors
  -Normalize pixel values to range [-1, 1]

5.Training Process:
  -Loss Function: CrossEntropyLoss
  -Optimizer: Adam
  -Learning Rate: 0.001
  -Batch Size: 64
  -Epochs: 15
  
6.Model Evaluation:
  -Model is evaluated on the test dataset
  -Accuracy is calculated using predicted vs true labels

7.Image Prediction:
The script supports single image prediction using a custom image:
  -Loads an external image
  -Applies same preprocessing
  -Outputs predicted CIFAR-10 class label
