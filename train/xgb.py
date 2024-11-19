import data_loader
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load data
X, y = data_loader.load_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the data
num_class = len(label_encoder.classes_)
print("Training XGBoost model...")
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
param = {'max_depth': 6, 'eta': 0.1, 'objective': 'multi:softmax', 'num_class': num_class}
num_round = 10
bst = xgb.train(param, dtrain, num_round)
print("Training complete.")

# Save the trained model
joblib.dump(bst, './Models/test/xgb_model.pkl')
print("Model saved.")

# Evaluate the model
y_pred = bst.predict(dtest).astype(int)  # Cast predictions to integer
report = classification_report(y_test, y_pred)

# Save the classification report
os.makedirs('./Reports', exist_ok=True)
with open('./Reports/xgb.txt', 'w') as f:
    f.write("XGBoost Classification Report:\n")
    f.write(report)

print("Classification report saved.")
