**Pneumonia Detection in Chest X-ray Images using Convolutional Neural Network and Pretrained Models**

This repository contains code and resources for detecting pneumonia in chest X-ray images using a Convolutional Neural Network (CNN) and pretrained models. The project aims to automate the process of pneumonia detection, assisting healthcare professionals in making accurate diagnoses.

**Dataset**

The dataset used for training and testing the models is not included in this repository due to its large size. However, it can be obtained from https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy/input. The dataset consists of chest X-ray images classified into two categories: pneumonia and normal. The dataset should be organized into the following structure:

```
dataset/
    train/
        pneumonia/
            pneumonia_image_1.jpg
            pneumonia_image_2.jpg
            ...
        normal/
            normal_image_1.jpg
            normal_image_2.jpg
            ...
    test/
        pneumonia/
            pneumonia_image_1.jpg
            pneumonia_image_2.jpg
            ...
        normal/
            normal_image_1.jpg
            normal_image_2.jpg
            ...
```

**Requirements**

To run the code in this repository, you need to have the following dependencies installed:
```
Python 
TensorFlow 
Keras 
NumPy 
OpenCV 
Matplotlib 
```
You can install the required packages using pip:
```
pip install tensorflow keras numpy opencv-python matplotlib
```
**Pretrained Models**

This project utilizes pretrained models to enhance the accuracy and efficiency of pneumonia detection. The following pretrained models are used:
```
ResNet50: A deep residual network with 50 layers.
VGG16: A 16-layer deep neural network.
```
**Usage**

To train the pneumonia detection model, follow these steps:

Download the dataset from https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy/input  and organize it as described in the Dataset section.
Adjust the configuration parameters in the config.py file, such as the dataset path, model selection, and training parameters.
Run the **Pnemonia_detection.ipynb** script to start the training process:
```
python Pnemonia_detection.ipynb
```
To test the trained model on new chest X-ray images, follow these steps:

Place the test images in the test_images/ directory.
The predictions will be displayed on the console, indicating whether each image is classified as pneumonia or normal.

**Results**

The performance of the pneumonia detection model can be evaluated based on various metrics, such as accuracy, precision, recall, and F1 score. The results of the trained model can be found in the results/ directory.

**Acknowledgements**

The dataset used in this project is sourced from https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy/input.
The pretrained models used are based on the works of [https://www.kaggle.com/code/shabhishek055/notebook866f354cce/edit].

**References**

[https://www.researchgate.net/publication/340961287_Pneumonia_Detection_Using_Convolutional_Neural_Networks_CNNs]
For any questions or inquiries, please contact [shabhishek055@gmail.com].

