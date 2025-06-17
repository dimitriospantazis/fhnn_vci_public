# This script loads the PLV matrices from .mat files in the 'phaseconnectivity_rois' directory,
# processes it, and saves it as a numpy array. It also extracts labels and saves them
# to a text file. Additionally, it computes the 95th percentile of the data to later use as cuttoff threshold, and plots a histogram.
# It assumes the directory structure is as follows:
# project_root/
# ├── data_matlab/
# │   └── phaseconnectivity_rois/
# │       ├── file1.mat
# │       ├── file2.mat
# │       └── ...
# └── data/
#     └── vci/


import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Get the current working directory
base_dir = os.getcwd()

# Construct the path to the 'phaseconnectivity_rois' directory (location of mat files)
file_dir = os.path.join(base_dir, 'data_matlab', 'phaseconnectivity_rois')

# List all mat files in the directory
mat_files = [f for f in os.listdir(file_dir) if f.endswith('.mat')]

# Initialize an numpy array to hold the data
data = np.zeros((len(mat_files), 90, 90))

# Loop through each mat file and load the data, use enumerate to get the index
for i, mat_file in enumerate(mat_files):
    # Construct the full file path
    file_path = os.path.join(file_dir, mat_file)
    
    # Load the data from the mat file
    try:
        mat_data = loadmat(file_path)
        # Extract the plv_rms data and assign it to the numpy array
        data[i] = mat_data['band'][2]['plv_rms'][0]
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")

# Create output directory
output_dir = os.path.join(base_dir, 'data', 'vci')
os.makedirs(output_dir, exist_ok=True)

# Save the data to a numpy file
output_file = os.path.join(output_dir, 'vci_plv_rms.npy')
np.save(output_file, data)

# Save labels to a text file
labels = mat_data['band'][2]['label'][0]
labels = [label[0] for label in labels]
labels_file = os.path.join(output_dir, 'vci_labels.txt')
with open(labels_file, 'w') as f:
    for label in labels:
        f.write(f"{label[0]}\n")

# Compute the 95 percentile values
#np.isnan(data).sum()  # Check for NaN values in the data
percentile_95  = np.nanpercentile(data, 95)  # Check the 95 percentile of the data
print(f"95th percentile: {percentile_95}")

# Save the 95th percentile to a text file
percentile_file = os.path.join(output_dir, 'vci_threshold.txt')
with open(percentile_file, 'w') as f:
    f.write(f"{percentile_95:.6f}\n")

# Plot histogram of all data values (excluding NaNs)
flattened_data = data[~np.isnan(data)].flatten()
plt.hist(flattened_data, bins=50, edgecolor='black')
plt.title('Histogram of PLV RMS Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

