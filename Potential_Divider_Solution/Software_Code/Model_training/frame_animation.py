import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# 配置输出目录
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# 数据路径
file_path = os.path.abspath('walking_cjj.csv')
data = pd.read_csv(file_path)

# 关节点映射及配色
col_prefix = {
    'left_foot': 'left_foot_index',
    'right_foot': 'right_foot_index',
    'left_knee': 'left_knee',
    'right_knee': 'right_knee',
    'left_hip': 'left_hip',
    'right_hip': 'right_hip',
    'left_shoulder': 'left_shoulder',
    'right_shoulder': 'right_shoulder',
    'head': 'head'
}
color_map = {
    'left_foot': 'deepskyblue', 'right_foot': 'steelblue',
    'left_knee': 'lightseagreen', 'right_knee': 'darkcyan',
    'left_hip': 'teal', 'right_hip': 'slategray',
    'left_shoulder': 'lightsteelblue', 'right_shoulder': 'dodgerblue',
    'head': 'dimgray'
}
connections = [
    ('left_hip','left_knee'),('left_knee','left_foot'),
    ('right_hip','right_knee'),('right_knee','right_foot'),
    ('left_hip','right_hip'),('left_shoulder','right_shoulder'),
    ('left_shoulder','head'),('right_shoulder','head')
]


start, end, step = 1400, 1630, 1
frames = list(range(start, end, step))


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=45, azim=270)  
ax.view_init(elev=0, azim=0)  

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

x_vals = data[[f'{col}_x' for col in col_prefix.values()]].values.flatten()
y_vals = -data[[f'{col}_y' for col in col_prefix.values()]].values.flatten()
z_vals = data[[f'{col}_z' for col in col_prefix.values()]].values.flatten()
ax.set_xlim(np.nanmin(x_vals), np.nanmax(x_vals))
ax.set_ylim(np.nanmin(y_vals), np.nanmax(y_vals))
ax.set_zlim(np.nanmin(z_vals), np.nanmax(z_vals))
ax.set_box_aspect([1,1,1])

# 初始化点和线对象
scatters = {j: ax.scatter([],[],[], color=color_map[j], s=50) for j in col_prefix}
lines = [ax.plot([],[],[], 'k-')[0] for _ in connections]


def update(i):
    row = data.iloc[frames[i]]
    joints = {}
    for j,prefix in col_prefix.items():
        x = row[f'{prefix}_x']
        y = -row[f'{prefix}_y']
        z = row[f'{prefix}_z']
        joints[j] = (x,y,z)
        scatters[j]._offsets3d = ([x],[y],[z])

    for idx,(a,b) in enumerate(connections):
        x1,y1,z1 = joints[a]
        x2,y2,z2 = joints[b]
        lines[idx].set_data([x1,x2],[y1,y2])
        lines[idx].set_3d_properties([z1,z2])

    return list(scatters.values()) + lines


fps = 10
t = 100  
anim = FuncAnimation(fig, update, frames=len(frames), interval=t, blit=True)
output = os.path.join(output_dir, 'simple_walking.mp4')
writer = FFMpegWriter(fps=fps)
anim.save(output, writer=writer)
print(f"Saved animation to {output}")
