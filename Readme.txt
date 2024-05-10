Neural Network Regression

Introduction
This repository contains Python code for implementing a simple neural network for regression tasks. The neural network is implemented from scratch using NumPy. It is trained and tested on the Boston housing dataset.

Requirements:
Python 3
pandas
numpy
matplotlib
Installation


You can install the required packages using pip:
pip install pandas numpy matplotlib


Description:
The code defines a neural network with two hidden layers.
The neural network is trained using backpropagation with ReLU activation function.
The Boston housing dataset is used for training and testing the model.
The training and testing errors are printed for each epoch.
Finally, a graph of testing error over epochs is plotted.

Dataset
The Boston housing dataset contains 506 samples with 13 features each. The target variable is the median value of owner-occupied homes in $1000s.

Credits
The code is adapted from various sources and tutorials on neural networks and backpropagation.