import data_loader
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load data
X, y = data_loader.load_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, './Models/scale/scaler.pkl')

# Train XGBoost model
print("Training XGBoost model...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)
print("Training complete.")

# Save the trained model
joblib.dump(xgb_model, './Models/test/xgb_model.pkl')
print("Model saved.")

# Evaluate the model
y_pred = xgb_model.predict(X_test_scaled)
report = classification_report(y_test, y_pred)

# Save the classification report
os.makedirs('./Reports', exist_ok=True)
with open('./Reports/xgb.txt', 'w') as f:
    f.write("XGBoost Classification Report:\n")
    f.write(report)

print("Classification report saved.")
