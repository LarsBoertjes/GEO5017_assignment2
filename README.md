# GEO5017_assignment2
Machine Learning Classification

## This program performs machine learning classification on point cloud data. It is designed for the GEO5017 course assignment 2.

All results can be reproduced by running the main.py file. The initial run will take +/- 15 minutes due to feature computation and hyperparameter tuning.

In order to run it the following python packages need to be installed:
- sklearn 
- seaborn
- numpy
- matplotlib
- tqdm
- scipy

The pointcloud data must be a folder in the same directory called 'data'.
Our code expects the .xyz files within the folder to be labeled as described in the GEO5017 Assignment 2 description. 

Our program writes the features to the text file 'data.txt' and reuses this file when rerunning the code.
Our program writes the hyperparameters to two text files 'RF_params.txt' & 'svm_params.txt', when you run it for the first time and reuses those files after. 
The hyperparameter ranges are the same as in the report and can be changed in the SVM and DF file.
While writing the 'svm_params.txt', it will give a warning due to the max iteration being reached which can be ignored. 

All figures are plotted but not automatically saved to a file.

A small description for each python file: 
| File name               | Content                                                                         |
|-------------------------|---------------------------------------------------------------------------------|
| bbox                    | Contains code from David Butterworth to compute the MOBB                        |
| DF                      | Contains all functions for RF hyperparameter tuning                             |
| eigen_features          | Contains all functions for the computation of eigenvalues features              |
| evaluation              | Contains the overlap matrix for evaluation of the features                      |
| extracting_features     | Contains the function to extract all features and write or read from a txt file |
| feature_selection       | Contains all functions for forward and backward search feature selection        |
| geometric_features      | Contains all functions for the computation of geomatric features                |
| learning_curve          | Contains all functions for plotting the learning curve                          |
| main                    | File to run, brings all files together                                          |
| plotting_features       | Contains almost all functions for plots concerning the features                 |
| read_data               | Contains the functions for reading the point clouds for feature extraction      |
| SVM                     | Contains all functions for SVM hyperparameter tuning                            |
| writing_hyperparameters | Contains the function to write the hyperparameters to a txt file                |
