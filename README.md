# CNN-for-CIFAR-10-Object-Localization
A deep learning project implementing a Convolutional Neural Network (CNN) for CIFAR-10 image classification and object localization. The model is trained using PyTorch to classify images and predict bounding box coordinates, combining classification and regression tasks for enhanced visual understanding


# Convolutional Neural Network - CIFAR-10 + Localization

## Overview
This project implements a CNN-based classifier for the CIFAR-10 dataset and includes a localization task formulated as a regression problem. The focus is on training a convolutional network using PyTorch, analyzing its performance, and extending it to predict object bounding boxes.

The project is divided into two main parts:
- **Training a convolutional neural network (CNN) to classify CIFAR-10 images.**
- **Extending the model to predict bounding box coordinates using regression.**

## Dataset
The dataset used is **CIFAR-10**, a widely used dataset consisting of **60,000 32x32 color images** in **10 classes** (such as airplanes, cars, birds, and more). The dataset is automatically downloaded and preprocessed for training, validation, and testing.

## Project Structure
### Data Loading & Preprocessing
- Downloads the CIFAR-10 dataset (if not already available).
- Extracts and loads the dataset.
- Applies transformations and normalization for training.
- Splits the dataset into training, validation, and testing sets.

### CNN Implementation in PyTorch
- Defines a custom convolutional network architecture.
- Uses batch normalization, dropout, and activation functions.
- Trains the CNN using cross-entropy loss and stochastic gradient descent (SGD).

### Model Training & Evaluation
- Trains the network and tracks performance over epochs.
- Evaluates classification accuracy on test data.
- Uses confusion matrices for model assessment.

### Object Localization (Regression Task)
- Modifies the network to predict bounding box coordinates.
- Uses Mean Squared Error (MSE) loss for regression.
- Evaluates localization performance with bounding box visualization.

## Installation & Requirements
To run this notebook, you need **Python 3** and the following dependencies:

```sh
pip install torch torchvision numpy matplotlib
```

## Usage
Clone the repository:

```sh
git clone https://github.com/LidanAvisar/CNN-for-CIFAR-10-Object-Localization
cd Convolutional-Network-CIFAR10
```

Open the Jupyter Notebook:

```sh
jupyter notebook "Convolutional Neural Network - Classifying CIFAR-10 + Localization as Regression.ipynb"
```

Run all cells to train and evaluate the CNN.

## Results & Analysis
The notebook visualizes:
- Sample images from the dataset.
- CNN decision boundaries.
- Loss and accuracy trends over training epochs.
- Predicted bounding boxes on test images.

