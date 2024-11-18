import joblib
import numpy as np
import warnings

# Paths to the saved model and scaler
MODEL_PATH = '../Models/test/knn_model.pkl'
SCALER_PATH = '../Models/scale/scaler.pkl'

def load_model_and_scaler():
    """Load the trained model and scaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        print("Model or scaler file not found. Ensure the files exist at the specified paths.")
        exit()

def get_user_input():
    """Prompt the user for input features."""
    print("Enter the features for prediction (separated by spaces):")
    try:
        input_data = list(map(float, input().split()))
        return np.array(input_data).reshape(1, -1)
    except ValueError:
        print("Invalid input. Please enter numeric values separated by spaces.")
        exit()

def main():
    # Load model and scaler
    warnings.filterwarnings("ignore", category=UserWarning)
    model, scaler = load_model_and_scaler()
    
    # Get user input
    input_data = get_user_input()
    
    # Scale input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the model
    prediction = model.predict(input_data_scaled)
    
    if(prediction[0]==1):
        print("Unhealthy")
    else: print("Healthy")

if __name__ == "__main__":
    main()
