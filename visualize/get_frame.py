"""
视频平均抽帧 & 网格拼接可视化工具。
对 MSRVTT/videos/video0.mp4 进行平均抽帧，将抽到的帧拼成网格大图保存。
"""

import cv2
import numpy as np
import os
import math

def average_sample_frames(video_path, num_frames):
    """
    对视频进行平均抽帧。

    Args:
        video_path: 视频文件路径。
        num_frames: 期望抽取的帧数。

    Returns:
        frames: 抽到的帧列表 (BGR numpy arrays)。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"视频帧数为 0: {video_path}")

    # 在 [0, total_frames) 区间均匀采样 num_frames 个位置
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def create_frame_grid(frames, target_size=(224, 224)):
    """
    将多帧图像水平拼接成一行大图。

    Args:
        frames: 帧列表 (BGR numpy arrays)。
        target_size: 每张小图的缩放尺寸 (宽, 高)。

    Returns:
        grid: 水平拼接后的图像。
    """
    n = len(frames)
    if n == 0:
        raise ValueError("帧列表为空，无法拼接")

    rows = 1
    cols = n

    # 统一缩放到 target_size
    resized = [cv2.resize(f, target_size, interpolation=cv2.INTER_AREA) for f in frames]

    # 创建黑色画布
    grid_h = rows * target_size[1]
    grid_w = cols * target_size[0]
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, frame in enumerate(resized):
        y1, y2 = 0, target_size[1]
        x1, x2 = i * target_size[0], (i + 1) * target_size[0]
        grid[y1:y2, x1:x2] = frame

    return grid

if __name__ == "__main__":
    # ==================== 用户可调参数 ====================
    VIDEO_PATH = "MSRVTT/videos/video0.mp4"     # 待处理视频
    OUTPUT_DIR = "visualize"                    # 输出目录
    NUM_FRAMES = 6                              # 抽帧数目（随意修改）
    # =====================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"正在从 {VIDEO_PATH} 抽取 {NUM_FRAMES} 帧...")
    frames = average_sample_frames(VIDEO_PATH, NUM_FRAMES)
    print(f"实际抽到 {len(frames)} 帧")

    grid = create_frame_grid(frames)
    output_path = os.path.join(OUTPUT_DIR, "video0_frames.jpg")
    cv2.imwrite(output_path, grid)

    print(f"已保存拼接图: {output_path}")
    print(f"  图像尺寸: {grid.shape[1]} x {grid.shape[0]} (宽 x 高)")
    print(f"  布局: 1 行 x {len(frames)} 列")