import data_loader
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, y_train = data_loader.load_data()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

model_path = './Models/test/knn_model.pkl'

scaler = joblib.load('./Models/scale/scaler.pkl') 

model = joblib.load(model_path)
X_test_scaled = scaler.transform(X_test)
print("Test Started")


y_pred = model.predict(X_test_scaled)
report = classification_report(y_test, y_pred)


with open('./Reports/knn.txt', 'w') as report_file:
    report_file.write("Classification Report:\n")
    report_file.write(report)

print("Report Saved")