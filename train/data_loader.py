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


normal_csv = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\Rotor-Healthy-Dataset.csv"
unbalace_forward_whirl = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\forward_whirl.csv"
unbalace_backward_whirl = r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\backward_whirl.csv"

def load_data():

    normal_data = pd.read_csv(normal_csv)
    unbalace_forward_whirl_data = pd.read_csv(unbalace_forward_whirl)
    unbalace_backward_whirl_data = pd.read_csv(unbalace_forward_whirl)
    

    normal_data['label'] = 0
    unbalace_forward_whirl_data['label'] = 1
    unbalace_backward_whirl_data['label'] = 2
    
    # Combine all data into a single DataFrame
    combined_data = pd.concat([normal_data, unbalace_forward_whirl_data, unbalace_backward_whirl_data], ignore_index=True)
    
    # Separate features (X) and labels (y)
    X = combined_data.drop('label', axis=1)
    y = combined_data['label']
    
    return X, y
