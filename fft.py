import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\COMSOL\Rotor-Unhealthy-Data.csv")  # Replace with your actual CSV file path

# Extract relevant columns
time = data['TIME(s)']
disk_2_y = data['Disk 2 Displacement Y component']
disk_2_z = data['Disk 2 Displacement Z component']

# Perform FFT on both components
fft_y = np.fft.fft(disk_2_y)
fft_z = np.fft.fft(disk_2_z)

# Calculate the phase of each FFT result
phase_y = np.angle(fft_y)
phase_z = np.angle(fft_z)

# Calculate the phase difference
phase_diff = phase_y - phase_z

# Classify as forward whirl (+ve phase diff) or backward whirl (-ve phase diff)
forward_whirl_mask = phase_diff > 0
backward_whirl_mask = phase_diff <= 0

# Prepare the DataFrames for forward whirl and backward whirl
forward_whirl_data = data[forward_whirl_mask].copy()
backward_whirl_data = data[backward_whirl_mask].copy()

# Save to CSVs
forward_whirl_data.to_csv('forward_whirl.csv', index=False)
backward_whirl_data.to_csv('backward_whirl.csv', index=False)

print("CSV files for forward and backward whirl have been saved.")
