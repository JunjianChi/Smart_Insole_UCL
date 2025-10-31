import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# 初始化RealSense的pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 创建滤波器
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

# 对齐深度帧到颜色帧
align_to = rs.stream.color
align = rs.align(align_to)

# 初始化MediaPipe Pose模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 关键节点索引（增加了脚上的所有点）
keypoints = {
    'head': mp_pose.PoseLandmark.NOSE,
    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP,
    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
    'left_heel': mp_pose.PoseLandmark.LEFT_HEEL,  # 脚部点
    'right_heel': mp_pose.PoseLandmark.RIGHT_HEEL, # 脚部点
    'left_foot_index': mp_pose.PoseLandmark.LEFT_FOOT_INDEX,  # 脚部点
    'right_foot_index': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX  # 脚部点
}

def get_depth_intrinsics(depth_frame):
    """ 获取深度图的内参 """
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    return depth_intrin

def convert_to_3d(depth_frame, intrinsics, x, y):
    """ 将2D坐标和深度信息转化为3D坐标 """
    depth = depth_frame.get_distance(x, y)
    if depth == 0:  # 如果深度为0，则认为该点无效
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)

try:
    while True:
        # 获取帧数据
        frames = pipeline.wait_for_frames()

        # 处理深度帧
        depth_frame = frames.get_depth_frame()
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        # 对齐深度帧到颜色帧
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 使用MediaPipe Pose进行姿态检测
        results = pose.process(rgb_image)

        # 如果检测到人体姿态
        if results.pose_landmarks:
            depth_intrinsics = get_depth_intrinsics(aligned_depth_frame)

            for keypoint_name, landmark_index in keypoints.items():
                landmark = results.pose_landmarks.landmark[landmark_index]
                x, y = int(landmark.x * color_image.shape[1]), int(landmark.y * color_image.shape[0])

                # 检查 (x, y) 是否在图像范围内
                if 0 <= x < color_image.shape[1] and 0 <= y < color_image.shape[0]:
                    # 将2D坐标转换为3D坐标
                    point_3d = convert_to_3d(aligned_depth_frame, depth_intrinsics, x, y)

                    if point_3d:
                        # 在图像上标记检测到的点
                        cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
                        # 显示3D坐标标签
                        label = f"{keypoint_name} ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})"
                        cv2.putText(color_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 200), 1)

        # 显示颜色图像
        cv2.imshow('Pose Detection with Depth', color_image)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止RealSense的pipeline
    pipeline.stop()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
