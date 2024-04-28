# DSCI-6011-Deep-Learning-Project
Welcome to our github project repository

# PROJECT - FOOD RECOGNITION SYSTEM

### This repository contains a description , and code for our project.
### Dataset - https://www.kaggle.com/datasets/dansbecker/food-101 (Downloaded from kaggle)
### Link to Google drive: https://drive.google.com/drive/folders/1h490vwPOpWXRAfo68be2gwFPrPnd7Cpg
As the dataset is very huge (10gb) we've chosen to take only 10 categories from them, each category consists of 1000 images: 
### apple_pie	baklava	beef_tartare  beignets baklava baby_back_ribs	beef_carpaccio beet_salad bibimbap omelette pizza

## Statement of project objectives:
The primary objective of this project is to develop a robust food recognition system that can accurately identify and classify images of food into one of 101 categories based on the Food-101 dataset.  The project aims to achieve high accuracy in food recognition by leveraging the capabilities of the Inception-v3 architecture. The objective is to train the model to accurately identify and classify food images with minimal error rates, ensuring reliable performance in real-world scenarios.

## Statement of value - why is this project worth doing?:
This project is worth doing because:
Health and Nutrition: It can significantly improve the ease and accuracy of tracking food intake, which is crucial for individuals monitoring their diet for health reasons.
Research and Development: Food recognition systems have applications beyond individual use, including research, public health initiatives, and food industry innovations. Researchers can leverage these systems to study dietary patterns, nutritional trends, and the impact of diet on health outcomes. 
Business Opportunities: From a business perspective, food recognition systems can create opportunities for companies to develop innovative products and services. This includes mobile apps, smart kitchen appliances, meal delivery services.

## Approach (i.e., what algorithms, datasets, models, tools, and techniques we intend to use to achieve the project objectives):
### Algorithm:
Convolutional Neural Networks (CNNs): The core of your model will be CNNs due to their excellence in handling image data. Specific architectures to consider include:
InceptionV3: Inception-v3 is a convolutional neural network (CNN) architecture that was developed by Google researchers as part of the Inception project, which aimed to improve the efficiency and performance of deep learning models for image classification and object recognition tasks. It is an evolution of the earlier Inception models, incorporating several key innovations to achieve better accuracy and computational efficiency. 

### Tecniques:
TensorFlow: It is extensively used for building and training deep learning models.
Matplotlib: Matplotlib is a plotting library for Python. It is used here for image visualization.
Convolutional Neural Networks (CNNs): The code defines a CNN architecture for image classification. It uses layers like Convolution2D,MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, and AveragePooling2D to build the model.
Data Augmentation: The code utilizes the ImageDataGenerator class from Keras for data augmentation, which helps in improving the generalization 

### Tools and technologies:
Python: As the primary programming language due to its wide support and libraries for machine learning.
TensorFlow : TensorFlow is an open-source machine learning framework developed and maintained by Google. It is one of the most popular and widely used libraries for building and training deep learning models. lt is a leading library for developing machine learning models.
OpenCV: Useful for handling image processing tasks outside of the deep learning model.
NumPy : For data manipulation.
Matplotlib:  For visualizing data distributions 
Jupyter Notebook or Google Colab: For interactive development and experimentation, particularly useful in the early stages of model design and testing.

### Dataset:
Food-101 Dataset: This dataset consists of 10,000 food images grouped into 10 categories. Each class contains 1000 images which provide a balanced dataset for training and testing.

### Deliverables: 
1.Trained Model:
The fully trained convolutional neural network model that can accurately recognize and classify images from the Food-101 dataset. This model is the core component of your project and directly addresses the objective of developing a robust food recognition system.
2.Codebase and Documentation:
Source Code: All code written for model development, including data preprocessing, model training, evaluation scripts, and any utility scripts for auxiliary tasks. The code will be well-commented and organized for easy understanding and reuse.
3.Documentation: Detailed documentation that includes:
System architecture and model design.
Description of the algorithms and technologies used.
How to train the model, perform predictions.

## Evaluation Methodology:
### Accuracy:
Accuracy is defined as the ratio of correct predictions to the total number of cases examined. For a food recognition system, it quantifies how often the model correctly identifies the category of food in an image. The formula for accuracy is:
Accuracy = Number of correct predictions / Total number of predictions
On the Test Set: After training and validation, evaluate the model’s accuracy on the unseen test set. This provides an unbiased evaluation of the model’s performance in the real world.

### Loss: 
Loss, in the context of machine learning and deep learning, represents a measure of how well a model's predictions match the true labels. It quantifies the difference between the predicted output and the actual target output. The goal during the training phase is to minimize this loss function, which leads to better performance of the model on unseen data.Cross-Entropy Loss : Used for multiclass classification tasks.Computes the average negative log likelihood of the true classes.

