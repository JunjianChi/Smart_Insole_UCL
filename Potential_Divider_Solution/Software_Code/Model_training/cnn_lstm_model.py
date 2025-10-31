import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from torch.utils.data import TensorDataset, DataLoader
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define file paths
# current_dir = os.getcwd()
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# file_path = os.path.join(parent_dir, 'dataset', 'walking', 'walking_cjj.csv')

file_path = os.path.abspath('walking_cjj.csv')


# ===========================
# Data Preprocessing Functions
# ===========================

def joint_preprocess(data):
    """
    Preprocess raw joints via removing outliers and applying gaussian filter.
    Outputs 6 joints with 3 coordinates each (total 18).
    """

    column_names = data.columns.tolist()
    column_names.remove('Timestamp')
    column_names.remove('Frame')
    column_names.remove('Matrix_0')
    column_names.remove('Matrix_1')
    column_names = [x for x in column_names if
                    'head' not in x and 'elbow' not in x and 'heel' not in x and 'shoulder' not in x]

    z_scores = data[column_names].apply(zscore)
    threshold = 2
    data_filtered = data[(z_scores.abs() < threshold).all(axis=1)]
    data_filtered = data_filtered.reset_index(drop=True)

    for column_name in column_names:
        data_filtered[column_name] = gaussian_filter(data_filtered[column_name], sigma=10)

    return data_filtered


def input_preprocess(data_filtered_joints):
    # Normalize the data
    data_matrix_0 = data_filtered_joints['Matrix_0']
    data_matrix_1 = data_filtered_joints['Matrix_1']

    matrix_0 = []
    matrix_1 = []

    # convert the data to matrix
    for data in data_matrix_0:
        temp = np.array(data.split(','), dtype=np.float32).reshape(33, 15)
        temp = temp[:, 2:-2]
        matrix_0.append(temp)

    for data in data_matrix_1:
        temp = np.array(data.split(','), dtype=np.float32).reshape(33, 15)
        temp = temp[:, 2:-2]
        matrix_1.append(temp)

    matrix_combined = []

    for frame_0, frame_1 in zip(matrix_0, matrix_1):
        combined_frame = np.hstack((frame_0, frame_1))
        matrix_combined.append(combined_frame)

    matrix_combined = np.array(matrix_combined)
    return matrix_0, matrix_1


def labelling(data_filtered_joints):
    """
    Outputs 6 joints with 3 coordinates each (total 18) relative to a ref point.
    """

    labels = []

    for i in range(len(data_filtered_joints)):
        frame_data = data_filtered_joints.iloc[i]

        # Extract hip coordinates
        left_hip = np.array([frame_data['left_hip_x'], frame_data['left_hip_y'], frame_data['left_hip_z']])
        right_hip = np.array([frame_data['right_hip_x'], frame_data['right_hip_y'], frame_data['right_hip_z']])
        mid_hip = (left_hip + right_hip) / 2  # Midpoint as reference

        # Extract other joints
        joints = {
            'left_knee': np.array([frame_data['left_knee_x'], frame_data['left_knee_y'], frame_data['left_knee_z']]),
            'right_knee': np.array(
                [frame_data['right_knee_x'], frame_data['right_knee_y'], frame_data['right_knee_z']]),
            'left_foot': np.array(
                [frame_data['left_foot_index_x'], frame_data['left_foot_index_y'], frame_data['left_foot_index_z']]),
            'right_foot': np.array(
                [frame_data['right_foot_index_x'], frame_data['right_foot_index_y'], frame_data['right_foot_index_z']]),
            'left_hip': left_hip,
            'right_hip': right_hip
        }

        # Calculate relative positions
        relative_positions = []
        for joint in joints.values():
            relative_position = joint - mid_hip
            relative_positions.extend(relative_position)  # Flatten

        labels.append(relative_positions)

    labels = np.array(labels)  # Shape: (num_frames, 18)
    print(f"Labels shape after preprocessing: {labels.shape}")  # Debugging

    # Min-Max Normalization to [-1, 1]
    data_min = labels.min(axis=0)
    data_max = labels.max(axis=0)
    range_ = data_max - data_min
    range_[range_ == 0] = 1  # Prevent division by zero
    labels_normalized = 2 * (labels - data_min) / range_ - 1

    return labels_normalized  # Shape: (num_frames, 18)



# ===========================
# Main program
# ===========================

# Load data
data = pd.read_csv(file_path)
data = data[100:7000]  # Select 5000 frames
data_original = data.copy()

# joints preprocess, returning 5000 frame without joint outliers
data_filtered_joints = joint_preprocess(data_original)

# input preprocess, returning 5000 frame combined matrix
matrix_l, matrix_r = input_preprocess(data_filtered_joints)

# labelling joints with calculating relative positions, returning labels
labels = labelling(data_filtered_joints)









