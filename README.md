# DSCI-6011-Deep-Learning-Project
Welcome to our github project repository

PROJECT - SALIENT OBJECT DETECTION

This repository contains a descropition, dataset - images , and code for our project.

Statement of project objectives:
The objective of the project is to identify the most "salient" or eye-catching objects in an image, which is a useful task in computer vision for improving image and video summarization, object recognition, and other tasks.

Statement of value - why is this project worth doing?:
This project is valuable because salient object detection can help computers understand what is most important in an image, which can be utilized in various applications such as image and video summarization, enhancing object recognition algorithms, and more, thereby improving the efficiency and effectiveness of computer vision tasks.

Approach (i.e., what algorithms, datasets, models, tools, and techniques we intend to use to achieve the project objectives):
Algorithm:
The core algorithm used in the project for salient object detection is not explicitly named as a traditional algorithm but is described in terms of the approach and techniques applied, such as maximizing contrast, minimizing entropy, etc. These algorithmic strategies aim to enhance the detection of salient objects within images.

Model:
DSSNet (Deeply Supervised Salient Object Detection Network): This model is a deep neural network architecture that combines the strengths of a Fully Convolutional Network (FCN) and a Recurrent Neural Network (RNN). The FCN component is responsible for extracting features from the image, while the RNN component captures spatial context and refines the predictions of salient objects. This hybrid approach allows DSSNet to efficiently and accurately identify salient objects in images, leveraging both local and global image features.

The DSSNet model, by integrating FCN and RNN, addresses the challenges of salient object detection through a deep learning framework, aiming to achieve high accuracy and efficiency in detecting the most visually striking elements of an image.

Dataset:
A specially curated and annotated dataset of 100 images from the University of New Haven, with manual annotations for salient objects and their boundaries.

Deliverables: 
Research Paper Reference: https://ieeexplore-ieee-org.unh-proxy01.newhaven.edu/document/8100181

Evaluation Methodology:
Precision: Precision, also known as positive predictive value, measures the accuracy of the detected objects. It is the ratio of correctly identified objects (true positives) to the total number of objects identified by the model (the sum of true positives and false positives). In simpler terms, it answers the question: "Of all the objects the model identified, how many were identified correctly?
" The formula for precision is: Precision=True Positives (TP)True Positives (TP)+False Positives (FP)Precision=True Positives (TP)+False Positives (FP)True Positives (TP)​

Recall:
Recall, also known as sensitivity, measures the model's ability to correctly identify all relevant objects in the image. It is the ratio of correctly identified objects (true positives) to the total number of actual objects in the image (the sum of true positives and false negatives). Recall answers the question: "Of all the actual objects present, how many did the model correctly identify?"
The formula for recall is: Recall=True Positives (TP)True Positives (TP)+False Negatives (FN)Recall=True Positives (TP)+False Negatives (FN)True Positives (TP)​

F-1 Score:
The F-1 score is the harmonic mean of precision and recall, offering a single metric that balances both. It is particularly useful when you need to compare two or more models that might have different precision and recall values. 
The F-1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is calculated as: 
F-1 Score=2×Precision×Recall/Precision+Recall


