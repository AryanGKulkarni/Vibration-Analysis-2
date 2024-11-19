import pandas as pd
import os
import random
import data_loader
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, y_train = data_loader.load_data()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("Training Started")

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
clf.fit(X_train, y_train)

print("Training Done")

import joblib

# Save the model to a file
joblib.dump(clf, './Models/test/rf_model.pkl')
print("Model Saved")

y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)


with open('./Reports/rf.txt', 'w') as report_file:
    report_file.write("Classification Report:\n")
    report_file.write(report)

print("Report Saved")