# Check_defect
How to classify defect/Healthy parts using computer vision and deep learning.
In this solution I am using transfer learning for classifying the Defect and Healthy parts.

Transfer Learning - Transfer learning is the application of knowledge gained from completing one task to help solve a different, but related, problem. Through transfer learning, methods are developed to transfer knowledge from one or more of these source tasks to improve learning in a related target task.

xception network -
Xception V1 model, with weights pre-trained on ImageNet.
On ImageNet, this model gets to a top-1 validation accuracy of 0.790 and a top-5 validation accuracy of 0.945.
Note that this model only supports the data format 'channels_last' (height, width, channels).
The default input size for this model is 299x299.

HOW TO RUN:
Every time when we run the model we should empty the model folder otherwise it throws the error of shape mismatch.
Model folder contains file after the execution:

1- Defect_healthy_classifier_model.h5

2- x_train.dat

3- y_train.dat

Steps :- 
1- Dowload the jupyter notebook and open the Check_defect.ipynb file in the localhost:8080 browser.

2- run the all cell in jupyter notebook.

3- check the accuracy,f1score,precision for the model at the end of the notebook.

4- I also have attached the python file.
