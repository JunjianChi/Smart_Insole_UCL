# import os
# import socket
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from threading import Thread
# import csv
# from datetime import datetime

# # -------------------- 配置 --------------------
# script_dir = os.path.dirname(os.path.abspath(__file__))
# csv_path   = os.path.join(script_dir, "insole_data_1.csv")

# # UDP Configuration
# UDP_IP      = "192.168.137.1"
# UDP_PORT    = 8999

# # Payload layout
# SENSOR_CNT   = 253
# IMU_CNT      = 6      # ax, ay, az, gx, gy, gz
# PAYLOAD_SIZE = SENSOR_CNT + IMU_CNT + 1  # +1 for foot side
# BUFFER_SIZE  = PAYLOAD_SIZE * 2          # bytes

# # Matrix dims
# MATRIX_ROW, MATRIX_COL = 33, 15

# # Foot index
# IDX_FOOT = 253
# IDX_AX   = 254
# IDX_AY   = 255
# IDX_AZ   = 256
# IDX_GX   = 257
# IDX_GY   = 258
# IDX_GZ   = 259

# # image_pattern and lookup_table (paste your full definitions here)

# # Lookup pattern and table (as before)
# image_pattern = np.array([
#     [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
#     [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
#     [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
#     [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
#     [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
#     [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
#     [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
#     [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
#     [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
#     [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
#     [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
#     [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
#     [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
#     [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
#     [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
#     [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
#     [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
#     [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
#     [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
#     [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
#     [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
#     [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
#     [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
# ])
# lookup_table = [
#     16,32,41,59,70,81,91,101,111,121,130,139,140,147,154,161,162,168,174,180,192,199,200,207,214,221,228,235,236,242,248,249,
#     4,24,33,50,60,71,82,92,102,112,122,131,132,141,148,155,156,163,169,175,186,193,194,201,208,215,222,229,230,237,243,244,
#     10,17,25,42,51,61,72,83,93,103,113,123,124,133,142,149,150,157,164,170,181,187,188,195,202,209,216,223,224,231,238,250,
#     5,11,18,34,43,52,62,73,84,94,104,114,115,125,134,143,144,151,158,165,176,182,183,189,196,203,210,217,218,225,232,251,
#     0,6,12,26,35,44,53,63,74,85,95,105,106,116,126,135,136,145,152,159,171,177,178,184,190,197,204,211,212,219,226,245,
#     1,2,7,19,27,36,45,54,64,75,86,96,97,107,117,127,128,137,146,153,166,172,173,179,185,191,198,205,206,213,220,239,
#     3,13,20,28,37,46,55,65,76,87,88,98,108,118,119,129,138,160,167,227,233,
#     8,14,21,29,38,47,56,66,77,78,89,99,109,110,120,234,240,
#     9,15,22,30,39,48,57,67,68,79,90,100,241,246,
#     23,31,40,49,58,69,80,247,252
# ]

# # -------------------- 打开 CSV 并写表头 --------------------
# # Columns: timestamp, left, right, ax, ay, az, gx, gy, gz
# csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["timestamp", "left", "right", "ax", "ay", "az", "gx", "gy", "gz"] )

# # -------------------- 数据容器 --------------------
# data_matrix_0 = np.zeros((MATRIX_ROW, MATRIX_COL), dtype=int)
# data_matrix_1 = np.zeros((MATRIX_ROW, MATRIX_COL), dtype=int)

# # -------------------- 绘图初始化 --------------------
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
# heatmap_0 = ax0.imshow(data_matrix_0, cmap='viridis', interpolation='nearest', vmin=0, vmax=4000)
# heatmap_1 = ax1.imshow(data_matrix_1, cmap='viridis', interpolation='nearest', vmin=0, vmax=4000)
# ax0.set_title("Left Insole");  fig.colorbar(heatmap_0, ax=ax0)
# ax1.set_title("Right Insole"); fig.colorbar(heatmap_1, ax=ax1)

# def update_heatmap(frame):
#     heatmap_0.set_data(data_matrix_0)
#     heatmap_1.set_data(data_matrix_1)
#     return heatmap_0, heatmap_1

# # -------------------- UDP 接收线程 --------------------
# def receive_data():
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.bind((UDP_IP, UDP_PORT))
#     print(f"Listening on UDP {UDP_IP}:{UDP_PORT}")
#     try:
#         while True:
#             data, _ = sock.recvfrom(BUFFER_SIZE)
#             payload = np.frombuffer(data, dtype=np.uint16)
#             if payload.size != PAYLOAD_SIZE:
#                 continue

#             # Extract sensors and IMU values
#             sensors = payload[:IDX_FOOT]
#             imu_vals = payload[IDX_AX:IDX_GZ+1]

#             # Rearrange sensor data into matrices
#             sorted_s = np.zeros_like(sensors, dtype=int)
#             for i, idx in enumerate(lookup_table):
#                 sorted_s[idx] = int(sensors[i])
#             inds = np.argwhere(image_pattern == 1)
#             client = int(payload[IDX_FOOT])
#             if client == 0:
#                 for (r,c), v in zip(inds, sorted_s):
#                     data_matrix_0[r,c] = v if v>100 else 0
#             else:
#                 for (r,c), v in zip(inds, sorted_s):
#                     data_matrix_1[r, MATRIX_COL-1-c] = v if v>200 else 0

#             # Timestamp
#             ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
#             # Serialize matrices: row comma-separated, rows semicolon-separated
#             left_str  = ";".join(",".join(str(x) for x in row) for row in data_matrix_0)
#             right_str = ";".join(",".join(str(x) for x in row) for row in data_matrix_1)

#             # Write CSV row with IMU columns
#             csv_writer.writerow([ts, left_str, right_str,
#                                  imu_vals[0], imu_vals[1], imu_vals[2],
#                                  imu_vals[3], imu_vals[4], imu_vals[5]])
#             csv_file.flush()

#     except KeyboardInterrupt:
#         print("Receiver stopping")
#     finally:
#         sock.close()
#         csv_file.close()

# # -------------------- 主流程 --------------------
# if __name__ == "__main__":
#     thread = Thread(target=receive_data, daemon=True)
#     thread.start()
#     ani = animation.FuncAnimation(
#         fig, update_heatmap,
#         interval=20, blit=True,
#         cache_frame_data=False
#     )
#     plt.show()





import os
import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import csv
from datetime import datetime

# -------------------- 配置 --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "insole_data_static_walking3.csv")

# UDP Configuration
UDP_IP      = "192.168.137.1"
UDP_PORT    = 8999

# Payload layout
SENSOR_CNT   = 253
IMU_CNT      = 6      # ax, ay, az, gx, gy, gz
PAYLOAD_SIZE = SENSOR_CNT + IMU_CNT + 1  # +1 for foot side
BUFFER_SIZE  = PAYLOAD_SIZE * 2          # bytes

# Matrix dims
MATRIX_ROW, MATRIX_COL = 33, 15

# Foot and IMU indices
IDX_FOOT = 253
IDX_AX   = 254
IDX_AY   = 255
IDX_AZ   = 256
IDX_GX   = 257
IDX_GY   = 258
IDX_GZ   = 259


# Lookup pattern and table (as before)
image_pattern = np.array([
    [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
])
lookup_table = [
    16,32,41,59,70,81,91,101,111,121,130,139,140,147,154,161,162,168,174,180,192,199,200,207,214,221,228,235,236,242,248,249,
    4,24,33,50,60,71,82,92,102,112,122,131,132,141,148,155,156,163,169,175,186,193,194,201,208,215,222,229,230,237,243,244,
    10,17,25,42,51,61,72,83,93,103,113,123,124,133,142,149,150,157,164,170,181,187,188,195,202,209,216,223,224,231,238,250,
    5,11,18,34,43,52,62,73,84,94,104,114,115,125,134,143,144,151,158,165,176,182,183,189,196,203,210,217,218,225,232,251,
    0,6,12,26,35,44,53,63,74,85,95,105,106,116,126,135,136,145,152,159,171,177,178,184,190,197,204,211,212,219,226,245,
    1,2,7,19,27,36,45,54,64,75,86,96,97,107,117,127,128,137,146,153,166,172,173,179,185,191,198,205,206,213,220,239,
    3,13,20,28,37,46,55,65,76,87,88,98,108,118,119,129,138,160,167,227,233,
    8,14,21,29,38,47,56,66,77,78,89,99,109,110,120,234,240,
    9,15,22,30,39,48,57,67,68,79,90,100,241,246,
    23,31,40,49,58,69,80,247,252
]

# -------------------- 打开 CSV 并写表头 --------------------
# Columns: timestamp, left, right, then left_ax..gz and right_ax..gz
csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
header = ["timestamp", "left", "right"]
# left IMU
header += [f"left_{k}" for k in ["ax","ay","az","gx","gy","gz"]]
# right IMU
header += [f"right_{k}" for k in ["ax","ay","az","gx","gy","gz"]]
csv_writer.writerow(header)

# -------------------- 数据容器 --------------------
data_matrix_0 = np.zeros((MATRIX_ROW, MATRIX_COL), dtype=int)
data_matrix_1 = np.zeros((MATRIX_ROW, MATRIX_COL), dtype=int)
# track latest IMU for left/right
imu_left  = [0]*6
imu_right = [0]*6

# -------------------- 绘图初始化 --------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
heatmap_0 = ax0.imshow(data_matrix_0, cmap='viridis', interpolation='nearest', vmin=0, vmax=4000)
heatmap_1 = ax1.imshow(data_matrix_1, cmap='viridis', interpolation='nearest', vmin=0, vmax=4000)
ax0.set_title("Left Insole");  fig.colorbar(heatmap_0, ax=ax0)
ax1.set_title("Right Insole"); fig.colorbar(heatmap_1, ax=ax1)

def update_heatmap(frame):
    heatmap_0.set_data(data_matrix_0)
    heatmap_1.set_data(data_matrix_1)
    return heatmap_0, heatmap_1

# -------------------- UDP 接收线程 --------------------
def receive_data():
    global imu_left, imu_right
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening on UDP {UDP_IP}:{UDP_PORT}")
    try:
        while True:
            data, _ = sock.recvfrom(BUFFER_SIZE)
            payload = np.frombuffer(data, dtype=np.uint16)
            if payload.size != PAYLOAD_SIZE:
                continue

            sensors = payload[:IDX_FOOT]
            imu_vals = payload[IDX_AX:IDX_GZ+1].tolist()
            client = payload[IDX_FOOT]

            # assign imu into left/right
            if client == 0:
                imu_left = imu_vals
            else:
                imu_right = imu_vals

            # Rearrange sensor to matrices
            sorted_s = np.zeros_like(sensors, dtype=int)
            for i, idx in enumerate(lookup_table):
                sorted_s[idx] = int(sensors[i])
            inds = np.argwhere(image_pattern == 1)
            if client == 0:
                for (r,c), v in zip(inds, sorted_s):
                    data_matrix_0[r,c] = v if v>100 else 0
            else:
                for (r,c), v in zip(inds, sorted_s):
                    data_matrix_1[r, MATRIX_COL-1-c] = v if v>200 else 0

            # Timestamp and serialize matrices
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            left_str  = ";".join(",".join(str(x) for x in row) for row in data_matrix_0)
            right_str = ";".join(",".join(str(x) for x in row) for row in data_matrix_1)

            # Write row
            row = [ts, left_str, right_str] + imu_left + imu_right
            csv_writer.writerow(row)
            csv_file.flush()

    except KeyboardInterrupt:
        print("Receiver stopping")
    finally:
        sock.close()
        csv_file.close()

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    thread = Thread(target=receive_data, daemon=True)
    thread.start()
    ani = animation.FuncAnimation(
        fig, update_heatmap,
        interval=20, blit=True,
        cache_frame_data=False
    )
    plt.show()
