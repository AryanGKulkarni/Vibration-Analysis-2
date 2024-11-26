import pandas as pd
import numpy as np

# Define column names
column_names = [
    "RPM",
    # "TIME(s)",
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


normal_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL2\Rotor Healthy Dataset.csv"
unbalance_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL2\Rotor Unhealthy Data.csv"
misalignment_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL2\Rotor-Misalignment-Data-Modified.csv"
misalignment_with_unbalance_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL2\Rotor-Misalignment-Unbalance-Modified.csv"



def load_data(corruption_fraction=0.1):
    # Load data from CSV files
    normal_data = pd.read_csv(normal_csv)
    unbalance_data = pd.read_csv(unbalance_csv)
    misalignment_data = pd.read_csv(misalignment_csv)
    misalignment_with_unbalance_data = pd.read_csv(misalignment_with_unbalance_csv)
    
    # Add labels to datasets
    normal_data['label'] = 0
    unbalance_data['label'] = 1
    misalignment_data['label'] = 2
    misalignment_with_unbalance_data['label'] = 3
    
    # Combine all data into a single DataFrame
    combined_data = pd.concat([normal_data, unbalance_data, misalignment_data, misalignment_with_unbalance_data], ignore_index=True)
    
    # Separate features (X) and labels (y)
    X = combined_data.drop('label', axis=1)
    y = combined_data['label']

    # Corrupt 10% of the feature data with random noise
    num_corrupted = int(X.shape[0] * corruption_fraction)
    indices_to_corrupt = np.random.choice(X.index, num_corrupted, replace=False)
    X.loc[indices_to_corrupt] += np.random.normal(0, 0.5, X.shape[1])  # Adjust noise magnitude as needed

    return X, y
