# GEO5017_assignment2
Machine Learning Classification

## This program performs machine learning classification on point cloud data. It is designed for the GEO5017 course assignment 2.

All results can be reproduced by running the main.py file.

In order to run it the following python packages need to be installed:
- sklearn 
- seaborn
- numpy
- matplotlib
- os
- bbox
- tqdm
- scipy

The pointcloud data must be a folder in the same directory called 'data'.
Our code expects the .xyz files within the folder to be labeled as described in the GEO5017 Assignment 2 description. 

Our program writes the hyperparameters to two text files 'RF_params.txt' & 'svm_params.txt', when you run it for the first time and reuses those files after. 
While writing the 'svm_params.txt', it will give a warning which can be ignored. 






