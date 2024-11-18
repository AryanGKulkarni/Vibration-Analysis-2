import pandas as pd
import os
import random
import data_loader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
X_train, y_train = data_loader.load_data()

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


print("Training Started")

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print("Training Done")

import joblib

# Save the model to a file
joblib.dump(gnb, './Models/test/gnb_model.pkl')
print("Model Saved")

y_pred = gnb.predict(X_test)
report = classification_report(y_test, y_pred)

with open('./Reports/gnb.txt', 'w') as report_file:
    report_file.write("Classification Report:\n")
    report_file.write(report)

print("Report Saved")