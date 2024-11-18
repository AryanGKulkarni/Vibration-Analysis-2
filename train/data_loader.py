import pandas as pd

# Define column names
column_names = [
    "RPM",
    "TIME(s)",
    "Bearing 1 Displacement field Y component",
    "Bearing 1 Displacement field Z component",
    "Disk 1 Displacement Y component",
    "Disk 1 Displacement Z component",
    "Disk 2 Displacement Y component",
    "Disk 2 Displacement Z component",
    "Disk 3 Displacement Y component",
    "Disk 3 Displacement Z component",
    "Bearing 2 Displacement field Y component",
    "Bearing 2 Displacement field Z component"
]

# Define the file paths
normal_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\Rotor-Healthy-Dataset.csv"
unhealthy_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\Rotor-Unhealthy-Data.csv"

def load_data():
    # Load the data from CSV files using the defined column names
    normal_data = pd.read_csv(normal_csv)
    unhealthy_data = pd.read_csv(unhealthy_csv)
    
    # Add labels to the data
    normal_data['label'] = 0  # Label 0 for normal condition
    unhealthy_data['label'] = 1  # Label 1 for unbalance fault
    
    # Combine all data into a single DataFrame
    combined_data = pd.concat([normal_data, unhealthy_data], ignore_index=True)
    
    # Separate features (X) and labels (y)
    X = combined_data.drop('label', axis=1)
    y = combined_data['label']
    
    return X, y
