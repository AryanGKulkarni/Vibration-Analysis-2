import pandas as pd
import os
import random
import data_loader
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
X_train, y_train = data_loader.load_data()

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training Started")

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, './Models/scale/scaler.pkl')

# Train the KNN model
knnmodel = KNeighborsClassifier(n_neighbors=5)
knnmodel.fit(X_train_scaled, y_train)

print("Training Done")

# Save the model
joblib.dump(knnmodel, './Models/test/knn_model.pkl')
print("Model Saved")

# Generate predictions and classification report
y_pred = knnmodel.predict(X_test_scaled)
report = classification_report(y_test, y_pred)


with open('./Reports/knn.txt', 'w') as report_file:
    report_file.write("Classification Report:\n")
    report_file.write(report)

print("Report Saved")
