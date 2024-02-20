# DLPROJECT
THIS IS MY PROJECT FOR DEEP LEARNING.
I have used a dataset containing RBC images of patients suffering from maleria as wel as normal patient's blood.
The medical image set is the dataset i have chosen and based on this dataset , this deep learning model will detect the presence of maleria or not in a given patient's body blood sample.

I have used python libraries and deep learning model (CNN) Convolutional Neural Networks, is a deep learning model.

This repository contains code for building a Convolutional Neural Network (CNN) model to detect malaria parasites in cell images. The model is trained on a dataset consisting of infected and uninfected cell images.
Dataset

The dataset used for training and testing the model is the Cell Images dataset, which consists of two main categories: infected and uninfected cells. The dataset is divided into training and testing sets.
Data Preprocessing

The cell images are loaded using OpenCV and PIL libraries.
Images are converted to grayscale and resized to 30x30 pixels.
Data augmentation techniques such as rotation, zoom, and horizontal flip can be applied for better model generalization.

Model Architecture

The CNN model architecture consists of the following layers:

   1. Convolutional layer with 30 filters and ReLU activation function.
   2. MaxPooling layer.
   3. Another Convolutional layer with 30 filters and ReLU activation function.
   4. MaxPooling layer.
   5. Flatten layer to convert 2D feature maps into a 1D feature vector.
   6. Dense (fully connected) layers with ReLU activation functions.
   7. Dropout layer to prevent overfitting.
   8. Output layer with a sigmoid activation function for binary classification.

The model is compiled using the Adam optimizer and binary cross-entropy loss function.


Training - The model is trained using the training data with a batch size of 50 and early stopping with a patience of 5 epochs to prevent overfitting. The training process is monitored using accuracy as the metric.
Evaluation

The trained model is evaluated using the testing data. The accuracy, precision, recall, and F1-score metrics are calculated. Additionally, a confusion matrix is generated to visualize the model's performance.
Results

The model achieves an accuracy of approximately 92.75% on the testing data. The precision, recall, and F1-score for both infected and uninfected classes are also provided.






DATASET LINK : https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
