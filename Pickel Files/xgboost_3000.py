# Importing required Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pickle

# Reading the dataset:
df = pd.read_csv('Zernike_Moments_YN_3000.csv', header=None)

# Shuffling the whole dataset:
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

# Converting categorical values of Target feature into numerical:
df[289].replace(['YES','NO'], [1,0], inplace=True)

# Splitting into Independent and Dependent features:
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-Test spilt:
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Creating an Object of XGBoost Classifier:
xgb = XGBClassifier()

# Fitting the Training data:
xgb.fit(x_train, y_train)

# Creating a pickle file for the classifier
filename = 'xgboost_3000.pkl'
pickle.dump(xgb, open(filename, 'wb'))