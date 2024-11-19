import pandas as pd

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


normal_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\Rotor-Healthy-Dataset.csv"
unbalance_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\Rotor-Unhealthy-Data.csv"
misalignment_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\Rotor-Misalignment-Dataset.csv"
misalignment_with_unbalance_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\Rotor-Misalignment-with-Unbalance.csv"


def load_data():

    normal_data = pd.read_csv(normal_csv, usecols=column_names)
    unbalance_data = pd.read_csv(unbalance_csv, usecols=column_names)
    misalignment_data = pd.read_csv(misalignment_csv, usecols=column_names)
    misalignment_with_unbalance_data = pd.read_csv(misalignment_with_unbalance_csv, usecols=column_names)
    

    normal_data['label'] = 0
    unbalance_data['label'] = 1
    misalignment_data['label'] = 2
    misalignment_with_unbalance_data['label'] = 3
    
    # Combine all data into a single DataFrame
    combined_data = pd.concat([normal_data, unbalance_data, misalignment_data, misalignment_with_unbalance_data], ignore_index=True)
    
    # Separate features (X) and labels (y)
    X = combined_data.drop('label', axis=1)
    y = combined_data['label']
    
    return X, y
