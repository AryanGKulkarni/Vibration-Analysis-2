import pandas as pd

# Define column names
column_names = [
    "Time",
    "Displacement Probe",
    "Angular Velocity Probe",
    "Acceleration Probe",
    "Displacement Amplitude X Probe",
    "Displacement Amplitude Y",
    "Displacement Amplitude Z",
    "Angular Velocity X Component Probe",
    "Angular Velocity Y Component Probe",
    "Angular Velocity Z Component Probe",
    "Acceleration X Probe",
    "Acceleration Y Probe",
    "Acceleration Z Probe"
]

# Define the file paths
normal_csv = r"Normal_path.csv"
unbalance_csv = r"Unbalance_path.csv"
misalignment_csv = r"Misalignment_path.csv"

def load_data():
    # Load the data from CSV files using the defined column names
    normal_data = pd.read_csv(normal_csv, names=column_names)
    unbalance_data = pd.read_csv(unbalance_csv, names=column_names)
    misalignment_data = pd.read_csv(misalignment_csv, names=column_names)
    
    # Add labels to the data
    normal_data['label'] = 0  # Label 0 for normal condition
    unbalance_data['label'] = 1  # Label 1 for unbalance fault
    misalignment_data['label'] = 2  # Label 2 for misalignment fault
    
    # Combine all data into a single DataFrame
    combined_data = pd.concat([normal_data, unbalance_data, misalignment_data], ignore_index=True)
    
    # Separate features (X) and labels (y)
    X = combined_data.drop('label', axis=1)
    y = combined_data['label']
    
    return X, y
