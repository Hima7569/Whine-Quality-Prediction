# Whine-Quality-Prediction
# Based upon the features the model predicts the quality of the winr 
# here the machine learning model was SVM
# Data Visualization
# Data Preprocessing
# Define Target Variable (y) and Feature Variables (X)
# Train Test Split
# Modeling
# Model Evaluation
# Predictions
# Explanation
# This code is a Python script that predicts the quality of white wine using Support Vector Machine (SVM) algorithm. 
# Data Import and Exploration: It begins by loading a dataset of white wine quality and exploring its structure using pandas functions like `describe()`, `info()`, and `head()`.
# Data Visualization:The script visualizes the relationship between 'fixed acidity' and 'volatile acidity' in the dataset using scatter and line plots.
# Data Preprocessing: It checks for missing values, performs basic statistics, and standardizes the feature variables using StandardScaler.
# Train-Test Split:The data is split into training and testing sets with an 80-20 ratio.
# Modeling:It employs a Support Vector Classifier (SVC) from scikit-learn to train the model on the training data.
# Model Evaluation:The script evaluates the trained model's performance on the testing data using confusion matrix and classification report.
# Predictions: Finally, it makes predictions on the testing data and prints them out.
