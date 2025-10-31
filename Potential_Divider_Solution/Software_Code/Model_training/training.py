import torch
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理函数
def remove_columns(matrix):
    return matrix[:, 2:13]  # 删除前面2列和后面2列，保留中间11列

def smooth_and_trim_pressure_matrix(pressure_matrix, sigma=1):
    smoothed_matrix = []
    for matrix in pressure_matrix:
        matrix = np.array(matrix.split(','), dtype=np.float32).reshape(33, 15)
        # smoothed_frame = gaussian_filter(matrix, sigma=sigma)
        trimmed_frame = remove_columns(matrix)
        smoothed_matrix.append(trimmed_frame)
    return np.array(smoothed_matrix)

# 自定义Dataset
class PressureDataset(Dataset):
    def __init__(self, pressure_inputs_seq_0, pressure_inputs_seq_1, labels_seq):
        assert len(pressure_inputs_seq_0) == len(pressure_inputs_seq_1) == len(labels_seq), \
            f"Dataset lengths are not equal: {len(pressure_inputs_seq_0)}, {len(pressure_inputs_seq_1)}, {len(labels_seq)}"
        self.pressure_inputs_seq_0 = pressure_inputs_seq_0
        self.pressure_inputs_seq_1 = pressure_inputs_seq_1
        self.labels_seq = labels_seq

    def __len__(self):
        return len(self.pressure_inputs_seq_0)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.pressure_inputs_seq_0[idx], dtype=torch.float32),
            torch.tensor(self.pressure_inputs_seq_1[idx], dtype=torch.float32),
            torch.tensor(self.labels_seq[idx], dtype=torch.float32)
        )

# 标签平滑函数
def moving_average(labels, window_size=5):
    smoothed_labels = np.copy(labels)
    for i in range(labels.shape[1]):
        smoothed_labels[:, i] = np.convolve(labels[:, i], np.ones(window_size)/window_size, mode='same')
    return smoothed_labels

def gaussian_smoothing(labels, sigma=1):
    smoothed_labels = gaussian_filter(labels, sigma=(0, sigma))
    return smoothed_labels

# 提取关节坐标并去除身高影响
def extract_relative_positions(data, height):
    labels = []
    xyz_values = []  # 用来存储所有关节的 x, y, z 坐标

    # 遍历数据并提取 x, y, z 坐标
    for i in range(len(data)):
        frame_data = data.iloc[i]

        # 提取髋关节坐标并计算中点
        left_hip = np.array([frame_data['left_hip_x'], frame_data['left_hip_y'], frame_data['left_hip_z']])
        right_hip = np.array([frame_data['right_hip_x'], frame_data['right_hip_y'], frame_data['right_hip_z']])
        left_shoulder = np.array([frame_data['left_shoulder_x'], frame_data['left_shoulder_y'], frame_data['left_shoulder_z']])
        right_shoulder = np.array([frame_data['right_shoulder_x'], frame_data['right_shoulder_y'], frame_data['right_shoulder_z']])
        mid_hip = (left_hip + right_hip) / 2  # 计算髋关节中点
        # mid hip , left shoulder, right shoulder rectangle center
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (mid_hip * 2 / 3 + mid_shoulder / 3)
        # mid_hip = mid_shoulder

        # 提取关节点坐标
        joints = {
            'left_knee': np.array([frame_data['left_knee_x'], frame_data['left_knee_y'], frame_data['left_knee_z']]),
            'right_knee': np.array([frame_data['right_knee_x'], frame_data['right_knee_y'], frame_data['right_knee_z']]),
            'left_foot': np.array([frame_data['left_foot_index_x'], frame_data['left_foot_index_y'], frame_data['left_foot_index_z']]),
            'right_foot': np.array([frame_data['right_foot_index_x'], frame_data['right_foot_index_y'], frame_data['right_foot_index_z']]),
            'left_hip': left_hip,
            'right_hip': right_hip
        }

        # 计算相对髋关节中点的位置并归一化，去除身高影响
        relative_positions = []
        for joint in joints.values():
            relative_position = (joint - mid_hip) / height  # 除以身高，去除身高影响
            relative_positions.extend(relative_position)
            xyz_values.append(relative_position)

        labels.append(relative_positions)

    labels = np.array(labels)
    xyz_values = np.array(xyz_values).reshape(-1, 3)

    # 使用 MinMaxScaler 对 x, y, z 坐标进行归一化
    scaler_xyz = MinMaxScaler()
    xyz_scaled = scaler_xyz.fit_transform(xyz_values)

    # 替换 labels 中的 x, y, z 坐标
    num_joints = len(joints)
    labels[:, 0::3] = xyz_scaled[:, 0].reshape(-1, num_joints)  # Replace x coordinates
    labels[:, 1::3] = xyz_scaled[:, 1].reshape(-1, num_joints)  # Replace y coordinates
    labels[:, 2::3] = xyz_scaled[:, 2].reshape(-1, num_joints)  # Replace z coordinates

    return labels, scaler_xyz

# 数据加载与预处理，去除身高影响
def load_data(file_path, height, timesteps=30):
    data = pd.read_csv(file_path)
    # data = data[100:5100].reset_index(drop=True)  # 5000帧数据
    data_length = len(data)
    total_data = 5500
    data = data.iloc[int(data_length / 2) - int(total_data / 2): int(data_length / 2) + int(total_data / 2)]

    # get the column names in list
    column_names = data.columns.tolist()
    column_names.remove('Timestamp')
    column_names.remove('Frame')
    column_names.remove('Matrix_0')
    column_names.remove('Matrix_1')
    # remove all head elbow heel
    column_names = [x for x in column_names if 'head' not in x and 'elbow' not in x and 'heel' not in x and 'shoulder' not in x]
    print(column_names)

    # remove outliers
    z_scores = data[column_names].apply(zscore)
    threshold = 3
    data = data[(z_scores.abs() < threshold).all(axis=1)]

    for column_name in column_names:
        data[column_name] = gaussian_filter(data[column_name], sigma=5)

    data_length = len(data)
    total_data = 5000
    data = data.iloc[int(data_length / 2) - int(total_data / 2): int(data_length / 2) + int(total_data / 2)]

    ####### Read Pressure Data #######
    data_matrix_0 = data['Matrix_0']
    data_matrix_1 = data['Matrix_1']

    matrix_0 = []
    matrix_1 = []

    # Convert the data to matrix
    for entry in data_matrix_0:
        temp = np.array(entry.split(','), dtype=np.float32).reshape(33, 15)
        temp = temp[:, 2:-2]
        matrix_0.append(temp)

    for entry in data_matrix_1:
        temp = np.array(entry.split(','), dtype=np.float32).reshape(33, 15)
        temp = temp[:, 2:-2]
        matrix_1.append(temp)

    # Copy the original matrix
    pressure_points_0 = np.array(matrix_0.copy())
    pressure_points_1 = np.array(matrix_1.copy())

    # Print the shape
    print(f"Matrix 0 shape: {pressure_points_0.shape}")
    print(f"Matrix 1 shape: {pressure_points_1.shape}")

    # # Gaussian filter (optional)
    # pressure_points_0 = gaussian_filter(matrix_0, sigma=1)
    # pressure_points_1 = gaussian_filter(matrix_1, sigma=1)
    
    # Scale the pressure points to 0-1
    scaler_input_0 = MinMaxScaler()
    scaler_input_1 = MinMaxScaler()
    pressure_inputs_scaled_0 = scaler_input_0.fit_transform(
        pressure_points_0.reshape(-1, 33 * 11)
    ).reshape(-1, 33, 11)
    pressure_inputs_scaled_1 = scaler_input_1.fit_transform(
        pressure_points_1.reshape(-1, 33 * 11)
    ).reshape(-1, 33, 11)

    pressure_inputs_seq_0, pressure_inputs_seq_1 = [], []
    for i in range(len(pressure_inputs_scaled_0) - timesteps + 1):
        pressure_inputs_seq_0.append(pressure_inputs_scaled_0[i:i + timesteps])
        pressure_inputs_seq_1.append(pressure_inputs_scaled_1[i:i + timesteps])

    pressure_inputs_seq_0 = np.array(pressure_inputs_seq_0)
    pressure_inputs_seq_1 = np.array(pressure_inputs_seq_1)

    # Process joint coordinates, remove height influence
    # Initialize filter data
    
    labels, scaler_xyz = extract_relative_positions(data, height)
    labels_seq = labels[timesteps - 1:]  # Align time steps

    # Ensure data and labels are the same length
    print(f"Loaded data from {file_path}")
    print(f"pressure_inputs_seq_0 shape: {pressure_inputs_seq_0.shape}")
    print(f"pressure_inputs_seq_1 shape: {pressure_inputs_seq_1.shape}")
    print(f"labels_seq shape: {labels_seq.shape}")

    assert len(pressure_inputs_seq_0) == len(pressure_inputs_seq_1) == len(labels_seq), \
        "pressure_inputs_seq and labels_seq must have the same length."

    return pressure_inputs_seq_0, pressure_inputs_seq_1, labels_seq


# 定义模型
class SeparatedCNN_LSTM_DNN(nn.Module):
    def __init__(self, timesteps, cnn_out_channels=64, lstm_hidden_size=128, lstm_layers=2, dnn_hidden_size=128):
        super(SeparatedCNN_LSTM_DNN, self).__init__()

        # 分别定义两个CNN
        self.cnn_0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 计算CNN输出的特征大小
        # 假设输入的每帧经过CNN后的尺寸为 [cnn_out_channels, 8, 5]
        self.cnn_output_size = cnn_out_channels * 8 * 5 * 2  # 两个CNN的输出拼接

        self.lstm = nn.LSTM(2048, lstm_hidden_size, num_layers=lstm_layers, batch_first=True)

        # DNN部分
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, dnn_hidden_size),
            nn.ReLU(),
            nn.Linear(dnn_hidden_size, 18),  # 输出12维标签 (4个关节，每个关节3个坐标)
        )

    def forward(self, x_0, x_1):
        batch_size, timesteps, h, w = x_0.size()

        # 分别通过两个CNN
        x_0 = x_0.view(-1, 1, h, w)  # 展开为 [batch_size * timesteps, 1, h, w]
        x_0 = self.cnn_0(x_0)

        x_1 = x_1.view(-1, 1, h, w)  # 展开为 [batch_size * timesteps, 1, h, w]
        x_1 = self.cnn_1(x_1)

        # 将 CNN 输出拉平
        x_0 = x_0.view(batch_size, timesteps, -1)
        x_1 = x_1.view(batch_size, timesteps, -1)
        x = torch.cat((x_0, x_1), dim=2)  # 在特征维度上拼接

        # 通过LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # 只取最后一个时间步的输出

        # 通过DNN
        x = self.fc(x)
        return x

# 训练和评估函数
def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs_0, inputs_1, labels in val_loader:
            inputs_0, inputs_1, labels = inputs_0.to(device), inputs_1.to(device), labels.to(device)
            outputs = model(inputs_0, inputs_1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs_0.size(0)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)

    # 计算每个输出维度的 MAE 和 R²
    mae_per_dim = mean_absolute_error(all_labels, all_outputs, multioutput='raw_values')
    r2_per_dim = r2_score(all_labels, all_outputs, multioutput='raw_values')

    return epoch_loss, mae_per_dim, r2_per_dim

# 早停类
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): 验证集损失在多少个epoch内没有下降时停止训练
            verbose (bool): 是否打印早停信息
            delta (float): 验证集损失下降的最小变化
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, save_path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, save_path)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, save_path)
            self.counter = 0

    def save_checkpoint(self, model, save_path):
        if self.verbose:
            print(f'Validation loss decreased. Saving model to {save_path}')
        torch.save(model.state_dict(), save_path)

# 训练模型并返回训练集的指标
def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, early_stopping, num_epochs=50):
    model.to(device)

    # 初始化变量来记录最后一个epoch的训练指标
    final_train_loss = 0.0
    final_train_mae = np.zeros(18)
    final_train_r2 = np.zeros(18)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_outputs = []

        for inputs_0, inputs_1, labels in train_loader:
            inputs_0, inputs_1, labels = inputs_0.to(device), inputs_1.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_0, inputs_1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs_0.size(0)

            all_train_labels.append(labels.detach().cpu().numpy())
            all_train_outputs.append(outputs.detach().cpu().numpy())

        # 计算训练集的损失、MAE和R^2
        train_loss = running_loss / len(train_loader.dataset)
        all_train_labels = np.vstack(all_train_labels)
        all_train_outputs = np.vstack(all_train_outputs)

        train_mae = mean_absolute_error(all_train_labels, all_train_outputs, multioutput='raw_values')
        train_r2 = r2_score(all_train_labels, all_train_outputs, multioutput='raw_values')

        # 记录最后一个epoch的训练指标
        final_train_loss = train_loss
        final_train_mae = train_mae
        final_train_r2 = train_r2

        # 验证集损失、MAE和R^2
        val_loss, val_mae_per_dim, val_r2_per_dim = evaluate_model(model, val_loader, criterion)

        # 打印训练集和验证集的损失、MAE和R^2
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Train MAE: {train_mae}, '
              f'Train R²: {train_r2}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val MAE: {val_mae_per_dim}, '
              f'Val R²: {val_r2_per_dim}')

        # 学习率调度器
        scheduler.step(val_loss)  # 根据验证损失调整学习率

        # 早停检查
        early_stopping(val_loss, model, 'checkpoint.pt')
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break

    # 加载最优模型
    model.load_state_dict(torch.load('checkpoint.pt'))

    return final_train_loss, final_train_mae, final_train_r2

# 可视化预测结果
def visualize_predictions(model, test_loader, smoothed=True):
    model.eval()  # 设置为评估模式
    all_labels = []
    all_outputs = []

    with torch.no_grad():  # 禁用梯度计算
        for inputs_0, inputs_1, true_labels in test_loader:
            inputs_0, inputs_1 = inputs_0.to(device), inputs_1.to(device)  # 数据迁移到GPU
            outputs = model(inputs_0, inputs_1)  # 模型预测

            # 将真实标签和模型输出都存储到列表中
            all_labels.append(true_labels.cpu().numpy())  # 从 GPU 回到 CPU
            all_outputs.append(outputs.cpu().numpy())

    # 合并所有批次的预测和标签
    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)

    if smoothed:
        # 对标签和输出进行平滑
        all_labels = moving_average(all_labels, window_size=5)
        all_outputs = moving_average(all_outputs, window_size=5)

    # 确保真实标签和模型输出有相同的维度
    assert all_labels.shape == all_outputs.shape, "真实值和预测值的形状不匹配!"

    return all_labels, all_outputs

# 主程序
if __name__ == '__main__':
    # 文件路径配置
    file_path = os.path.abspath('walking_cjj.csv')
    data = pd.read_csv(file_path)

    # 每个文件对应的人的身高（以米为单位）
    heights = [1.70]
    # 仅第一个人的身高
    # heights = [1.70, 1.73, 1.72, 1.75, 1.80]

    # 定义TimeSeriesSplit
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 定义标签平滑方法
    use_moving_average = True  # 如果使用高斯平滑，则设为False
    window_size = 5  # 移动平均窗口大小
    sigma = 1  # 高斯平滑标准差

    # 初始化模型参数
    timesteps = 75
    criterion = nn.SmoothL1Loss()  # Huber loss
    lr = 0.0001

    # 创建模型保存目录
    model_dir = 'saved_models_single_person'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 仅训练第一个人
   

    # 加载并处理该个人的数据
    pressure_inputs_seq_0, pressure_inputs_seq_1, labels_seq = load_data(file_path, heights, timesteps=timesteps)

    # 标签平滑（选择移动平均或高斯平滑）
    if use_moving_average:
        labels_seq = moving_average(labels_seq, window_size=window_size)
        print(f"Applied Moving Average smoothing with window size {window_size} to labels.")
    else:
        labels_seq = gaussian_smoothing(labels_seq, sigma=sigma)
        print(f"Applied Gaussian smoothing with sigma {sigma} to labels.")

    # 确保数据和标签长度一致
    assert len(pressure_inputs_seq_0) == len(pressure_inputs_seq_1) == len(labels_seq), \
        "pressure_inputs_seq and labels_seq must have the same length."

    # 定义TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 初始化存储指标的列表
    train_losses = []
    train_maes = []
    train_r2s = []

    val_losses = []
    val_maes = []
    val_r2s = []

    # 初始化存储所有折叠的预测结果
    all_val_labels = []
    all_val_outputs = []

    fold = 1
    for train_index, val_index in tscv.split(pressure_inputs_seq_0):
        print(f"\n===== Fold {fold}/{n_splits} =====")

        # 划分训练集和验证集
        X_train_0, X_val_0 = pressure_inputs_seq_0[train_index], pressure_inputs_seq_0[val_index]
        X_train_1, X_val_1 = pressure_inputs_seq_1[train_index], pressure_inputs_seq_1[val_index]
        y_train, y_val = labels_seq[train_index], labels_seq[val_index]

        # 打印划分信息
        print(f"Train size: {X_train_0.shape[0]}, Val size: {X_val_0.shape[0]}")

        # 对训练集和验证集进行独立的归一化
        scaler_input_0 = MinMaxScaler()
        scaler_input_1 = MinMaxScaler()

        # 归一化训练集
        X_train_0_scaled = scaler_input_0.fit_transform(X_train_0.reshape(-1, 33 * 11)).reshape(X_train_0.shape[0], timesteps, 33, 11)
        X_val_0_scaled = scaler_input_0.transform(X_val_0.reshape(-1, 33 * 11)).reshape(X_val_0.shape[0], timesteps, 33, 11)

        X_train_1_scaled = scaler_input_1.fit_transform(X_train_1.reshape(-1, 33 * 11)).reshape(X_train_1.shape[0], timesteps, 33, 11)
        X_val_1_scaled = scaler_input_1.transform(X_val_1.reshape(-1, 33 * 11)).reshape(X_val_1.shape[0], timesteps, 33, 11)

        # 创建训练集和验证集的 Dataset 和 DataLoader
        train_dataset = PressureDataset(X_train_0_scaled, X_train_1_scaled, y_train)
        val_dataset = PressureDataset(X_val_0_scaled, X_val_1_scaled, y_val)

        # 打印数据集大小
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 初始化模型、优化器和学习率调度器
        model = SeparatedCNN_LSTM_DNN(timesteps=timesteps).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )

        # 初始化早停
        early_stopping = EarlyStopping(patience=20, verbose=True)

        # 训练模型并在每个 epoch 后验证
        print(f"Training Fold {fold}...")
        train_loss, train_mae, train_r2 = train_model(
            train_loader, val_loader, model, criterion, optimizer, scheduler, early_stopping, num_epochs=100
        )

        # 记录训练集的指标
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        train_r2s.append(train_r2)

        # 加载最优模型
        best_model_path = os.path.join(model_dir, f'fold_{fold}_best_model.pth')
        model.load_state_dict(torch.load('checkpoint.pt'))
        torch.save(model.state_dict(), best_model_path)

        # 评估验证集并记录指标
        val_loss, val_mae_per_dim, val_r2_per_dim = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_maes.append(val_mae_per_dim)
        val_r2s.append(val_r2_per_dim)

        print(f"Fold {fold} 完成 - Train R²: {train_r2}, Val R²: {val_r2_per_dim}")

        # 收集所有折叠的预测结果
        labels, outputs = visualize_predictions(model, val_loader, smoothed=use_moving_average)
        all_val_labels.append(labels)
        all_val_outputs.append(outputs)

        fold += 1  # 进入下一个折

        # 训练完成后，计算并打印汇总指标
    print("\n===== 交叉验证结果汇总 =====")
    print(f"训练集 R²: 平均 = {np.mean(train_r2s, axis=0)}, 标准差 = {np.std(train_r2s, axis=0)}")
    print(f"验证集 R²: 平均 = {np.mean(val_r2s, axis=0)}, 标准差 = {np.std(val_r2s, axis=0)}")

    print("\n详细每折的 R² 值：")


    final_model_path = os.path.join(model_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    # 保存归一化器
    joblib.dump(scaler_input_0, os.path.join(model_dir, 'scaler_input_0.pkl'))
    joblib.dump(scaler_input_1, os.path.join(model_dir, 'scaler_input_1.pkl'))
    joblib.dump(scaler_xyz,       os.path.join(model_dir, 'scaler_xyz.pkl'))
    print("Saved all scalers.")

    joints_names = ['left_knee', 'right_knee', 'left_foot', 'right_foot', 'left_hip', 'right_hip']
    coordinates = ['X', 'Y', 'Z']

    for i in range(n_splits):
        print(f"折 {i+1}:")
        for dim in range(18):
            joint = dim // 3 + 1  # 0-2: joint1, 3-5: joint2, etc.
            axis = dim % 3  # 0: X, 1: Y, 2: Z
            axis_str = ['X', 'Y', 'Z'][axis]
            print(f"  Joint: {joints_names[joint-1]} {axis_str}: {val_r2s[i][dim]:.4f}")

    # 生成每个关节点每个轴的评估参数表格
    metrics_table = pd.DataFrame(columns=['Joint', 'Axis', 'R² Mean', 'R² Std', 'MAE Mean', 'MAE Std'])
    
    try:
        for dim in range(18):
            joint = joints_names[dim // 3]
            axis = coordinates[dim % 3]
            r2_mean = np.mean([fold_r2[dim] for fold_r2 in val_r2s])
            r2_std = np.std([fold_r2[dim] for fold_r2 in val_r2s])
            mae_mean = np.mean([fold_mae[dim] for fold_mae in val_maes])
            mae_std = np.std([fold_mae[dim] for fold_mae in val_maes])
            # metrics_table = metrics_table.append({
            #     'Joint': joint,
            #     'Axis': axis,
            #     'R² Mean': r2_mean,
            #     'R² Std': r2_std,
            #     'MAE Mean': mae_mean,
            #     'MAE Std': mae_std
            # }, ignore_index=True)
            # DataFrame do not have append method, use loc instead
            metrics_table.loc[dim] = [joint, axis, r2_mean, r2_std, mae_mean, mae_std]
    except Exception as e:
        print(e)

    print("\n===== 每个关节点每个轴的评估参数 =====")
    print(metrics_table)

    # 可视化所有预测结果与真实值
    all_val_labels = np.vstack(all_val_labels)
    all_val_outputs = np.vstack(all_val_outputs)

    if use_moving_average:
        all_val_labels = moving_average(all_val_labels, window_size=window_size)
        all_val_outputs = moving_average(all_val_outputs, window_size=window_size)

    # 计算每个节点的绝对误差
    errors = np.abs(all_val_labels - all_val_outputs)
    overall_mae = np.mean(errors)
    overall_mse = mean_squared_error(all_val_labels, all_val_outputs)
    overall_rmse = np.sqrt(overall_mse)
    print(f'Overall MAE: {overall_mae:.4f}')
    print(f'Overall RMSE: {overall_rmse:.4f}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()


    # 可视化四个关节点的预测与真实值
    # Plot the predictions and true labels in validation data
    plt.figure(figsize=(20, 15))

    for i in range(6):  # Six joints
        for j in range(3):  # Three coordinates
            dim = i * 3 + j
            plt.subplot(6, 3, dim + 1)
            plt.plot(all_val_labels[:, dim], label='True', alpha=0.7)
            plt.plot(all_val_outputs[:, dim], label='Predicted', linestyle='dashed', alpha=0.7)
            plt.title(f'{joints_names[i]} {coordinates[j]}')
            plt.legend()

    plt.tight_layout()
    plt.show()