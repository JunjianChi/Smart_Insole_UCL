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

    # 遍历每帧，分别从 matrix_0 和 matrix_1 中获取对应帧，并进行列拼接
    for frame_0, frame_1 in zip(matrix_0, matrix_1):
        # 将 frame_0 和 frame_1 按列拼接
        combined_frame = np.hstack((frame_0, frame_1))
        matrix_combined.append(combined_frame)

    matrix_combined = np.array(matrix_combined)
    print("Matrix combined shape",matrix_combined.shape)
    return matrix_combined


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

    # 创建子图，每个节点绘制一张图
    # num_frames = labels_normalized.shape[0]  # 5000 帧
    # num_joints = labels_normalized.shape[1]  # 18 个节点
    #
    # fig, axes = plt.subplots(num_joints, 1, figsize=(10, num_joints * 3))
    #
    # # 绘制每个节点在 5000 帧上的值
    # for i in range(num_joints):
    #     axes[i].plot(range(num_frames), labels_normalized[:, i])
    #     axes[i].set_title(f'Joint {i + 1}')
    #     axes[i].set_xlabel('Frame')
    #     axes[i].set_ylabel('Normalized Value')
    #
    # # 调整布局
    # plt.tight_layout()
    # plt.show()

    return labels_normalized  # Shape: (num_frames, 18)


# ===========================
# Deep Learning Model
# ===========================

class C3D_LSTM(nn.Module):

    def __init__(self, height=33, width=22, feature_size=(1,1), num_joints=6, in_channel=1):

        '''
        :param height: height of pressure matrix (33)
        :param width: width of pressure matrix (22)
        # :param sample_duration: frames
        :param feature_size: expected input size of LSTM
        :param num_joints: number of joints (x,y,z)
        :param in_channel: channel number
        '''

        super(C3D_LSTM, self).__init__()
        self.height = height
        self.width = width
        self.feature_size = feature_size  # Desired spatial output size after C3D
        self.in_channel = in_channel
        self.lstm_hidden_size = 256

        self.group1 = nn.Sequential(
            nn.Conv3d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # down-sample
            nn.Dropout(0.3))

        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout(0.3))

        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1,  padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3,  stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout(0.3))

        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3,  stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3,  stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout(0.3))

        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3,  stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3,  stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Dropout(0.3))

        last_height = 2
        last_width = 1
        # self.reduce_dim = nn.Linear(512 * last_height * last_width, 256)

        self.lstm = nn.LSTM(
            input_size=512 * last_height * last_width,  # last_size is small, so this reduces input_size
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_joints * 3)
        )

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.group1(x)
        # print(f"After group1: {x.shape}")
        x = self.group2(x)
        # print(f"After group2: {x.shape}")
        x = self.group3(x)
        # print(f"After group3: {x.shape}")
        x = self.group4(x)
        # print(f"After group4: {x.shape}")
        x = self.group5(x)
        # print(f"After group5: {x.shape}")

        # Reshape for LSTM
        batch_size, channels, frames, height, width = x.size()
        x = x.view(batch_size, frames, -1)  # LSTM expects input of shape (batch, seq_len, input_size)
        # print(f"Before LSTM: {x.shape}")

        x, _ = self.lstm(x)
        # print(f"After LSTM: {x.shape}")

        # 全连接层
        x = self.fc(x)
        print(f"After FC: {x.shape}")

        return x


# ===========================
# Dataset split for time sequence
# ===========================

sequence_length = 50
stride = 5

def create_sequences(data, labels, sequence_length, stride):
    sequences = []
    sequence_labels = []

    for i in range(0, len(data) - sequence_length + 1, stride):
        sequences.append(data[i:i + sequence_length])
        sequence_labels.append(labels[i:i + sequence_length])

    return np.array(sequences), np.array(sequence_labels)

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
matrix_combined = input_preprocess(data_filtered_joints)

# labelling joints with calculating relative positions, returning labels
labels = labelling(data_filtered_joints)

# Splitting into train and val set via time proportion
sample_duration = len(labels)
split_ratio = 0.8
split_index = int(sample_duration* split_ratio)
train_matrix = matrix_combined[:split_index]
train_labels = labels[:split_index]
val_matrix = matrix_combined[split_index:]
val_labels = labels[split_index:]

# Split by time sequence for LSTM model
train_sequences, train_sequence_labels = create_sequences(train_matrix, train_labels, sequence_length, stride)
val_sequences, val_sequence_labels = create_sequences(val_matrix, val_labels, sequence_length, stride)
train_sequences_tensor = torch.tensor(train_sequences, dtype=torch.float32)
train_sequence_labels_tensor = torch.tensor(train_sequence_labels, dtype=torch.float32)
val_sequences_tensor = torch.tensor(val_sequences, dtype=torch.float32)
val_sequence_labels_tensor = torch.tensor(val_sequence_labels, dtype=torch.float32)
train_sequences_tensor = train_sequences_tensor.unsqueeze(1)  # (3901, 1, 100, 33, 22)
val_sequences_tensor = val_sequences_tensor.unsqueeze(1)      # (901, 1, 100, 33, 22)

# Create dataset and loader
batch_size = 64
train_dataset = TensorDataset(train_sequences_tensor, train_sequence_labels_tensor)
val_dataset = TensorDataset(val_sequences_tensor, val_sequence_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# 添加通道维度

# Train model
model = C3D_LSTM(height=33, width=22, num_joints=6, in_channel=1)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

patience = 30  # Number of epochs to wait before stopping
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

num_epochs = 50  # Maximum number of epochs
train_losses = []
val_losses = []
train_metrics = []
val_metrics = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')

    # -----------------------------
    # Training Phase
    # -----------------------------
    model.train()
    running_loss = 0.0
    train_mse_list = []
    train_r2_list = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Compute batch-wise metrics directly
        outputs_cpu = outputs.detach().cpu().numpy().reshape(-1, 18)
        labels_cpu = labels.cpu().numpy().reshape(-1, 18)

        # Calculate metrics for this batch
        train_mse = mean_squared_error(labels_cpu, outputs_cpu, multioutput='raw_values')
        train_r2 = r2_score(labels_cpu, outputs_cpu, multioutput='raw_values')

        # Collect metrics for each batch
        train_mse_list.append(train_mse)
        train_r2_list.append(train_r2)

    # Calculate average loss and metrics for this epoch
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)

    # Compute the mean MSE and R2 across all batches
    epoch_mse = np.mean(train_mse_list, axis=0)
    epoch_r2 = np.mean(train_r2_list, axis=0)
    train_metrics.append((epoch_mse, epoch_r2))

    print(f'Training Loss: {epoch_loss:.4f}')
    print('Training Metrics per Output Dimension:')
    for i in range(18):
        print(f'  Output {i + 1}: MSE={epoch_mse[i]:.4f}, R2={epoch_r2[i]:.4f}')

    # -----------------------------
    # Validation Phase
    # -----------------------------
    model.eval()
    val_running_loss = 0.0
    val_mse_list = []
    val_r2_list = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)

            # Compute batch-wise metrics directly
            outputs_cpu = outputs.cpu().numpy().reshape(-1, 18)
            labels_cpu = labels.cpu().numpy().reshape(-1, 18)

            # Calculate metrics for this batch
            val_mse = mean_squared_error(labels_cpu, outputs_cpu, multioutput='raw_values')
            val_r2 = r2_score(labels_cpu, outputs_cpu, multioutput='raw_values')

            # Collect metrics for each batch
            val_mse_list.append(val_mse)
            val_r2_list.append(val_r2)

    # Calculate average validation loss
    val_epoch_loss = val_running_loss / len(val_dataset)
    val_losses.append(val_epoch_loss)

    # Compute the mean MSE and R2 across all validation batches
    val_epoch_mse = np.mean(val_mse_list, axis=0)
    val_epoch_r2 = np.mean(val_r2_list, axis=0)
    val_metrics.append((val_epoch_mse, val_epoch_r2))

    print(f'Validation Loss: {val_epoch_loss:.4f}')
    print('Validation Metrics per Output Dimension:')
    for i in range(18):
        print(f'  Output {i + 1}: MSE={val_epoch_mse[i]:.4f}, R2={val_epoch_r2[i]:.4f}')

    # -----------------------------
    # Early Stopping Check (optional)
    # -----------------------------
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping!')
            break

# 8. Load Best Model Weights
model.load_state_dict(best_model_wts)

# 9. Visualize Predictions
# Collect predictions on the validation set
model.eval()
val_running_outputs = []
val_running_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        val_running_outputs.append(outputs.cpu())
        val_running_labels.append(labels.cpu())

# Concatenate all outputs and labels
val_outputs = torch.cat(val_running_outputs, dim=0).numpy().reshape(-1, 18)
val_labels = torch.cat(val_running_labels, dim=0).numpy().reshape(-1, 18)

# Plot true vs. predicted values for each output dimension
num_outputs = 18
fig, axs = plt.subplots(6, 3, figsize=(15, 20))
axs = axs.ravel()

for i in range(num_outputs):
    axs[i].plot(val_labels[:, i], label='True')
    axs[i].plot(val_outputs[:, i], label='Predicted')
    axs[i].set_title(f'Output Dimension {i + 1}')
    axs[i].legend()

plt.tight_layout()
plt.show()