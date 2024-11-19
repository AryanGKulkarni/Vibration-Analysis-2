from sklearn import svm
import data_loader
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

X_train, y_train = data_loader.load_data()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("Training Started")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, './Models/scaler.pkl')

svmmodel = svm.SVC()
svmmodel.fit(X_train_scaled, y_train)

print("Training Done")


# Save the model to a file
joblib.dump(svmmodel, './Models/test/svm_model.pkl')
print("Model Saved")

y_pred = svmmodel.predict(X_test)
report = classification_report(y_test, y_pred)


with open('./Reports/svm.txt', 'w') as report_file:
    report_file.write("Classification Report:\n")
    report_file.write(report)

print("Report Saved")