import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def load_npz_data(file_path):
    try:
        with np.load(file_path) as data:
            keypoints2d = data['keypoints2d']
            right = data['right']
            return keypoints2d, right
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def analyze_wrist_movement(directory):
    files = sorted([f for f in Path(directory).glob("*.npz")], key=natural_sort_key)
    print(f"Total files found: {len(files)}")
    
    data = {
        'left_wrist': [],
        'right_wrist': [],
        'frame_numbers': {},  # 使用字典分别存储左右手的帧号
        'left_frames': [],
        'right_frames': []
    }
    
    for idx, file_path in enumerate(files):
        keypoints2d, right = load_npz_data(file_path)
        if keypoints2d is None:
            continue
        
        if keypoints2d.shape[1] == 21:  # 单手数据
            if right[0] == 0:  # 左手
                data['left_wrist'].append(keypoints2d[0, 0])
                data['left_frames'].append(idx)
            else:  # 右手
                data['right_wrist'].append(keypoints2d[0, 0])
                data['right_frames'].append(idx)
        else:  # 双手数据 (42个点)
            data['left_wrist'].append(keypoints2d[0, 0])
            data['right_wrist'].append(keypoints2d[0, 21])
            data['left_frames'].append(idx)
            data['right_frames'].append(idx)
    
    # 转换为numpy数组
    data['left_wrist'] = np.array(data['left_wrist']) if data['left_wrist'] else np.array([])
    data['right_wrist'] = np.array(data['right_wrist']) if data['right_wrist'] else np.array([])
    data['left_frames'] = np.array(data['left_frames'])
    data['right_frames'] = np.array(data['right_frames'])
    
    print(f"Processed frames: {len(files)}")
    print(f"Left wrist data points: {len(data['left_wrist'])}")
    print(f"Right wrist data points: {len(data['right_wrist'])}")
    
    return data

def plot_wrist_trajectories(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Wrist Movement Analysis', fontsize=16)
    
    # 绘制X轴变化
    if len(data['left_wrist']) > 0:
        left_gaps = np.where(np.diff(data['left_frames']) > 1)[0]
        prev_idx = 0
        for gap in left_gaps:
            # 绘制连续段 - X坐标
            ax1.plot(data['left_frames'][prev_idx:gap+1], 
                    data['left_wrist'][prev_idx:gap+1, 0],  # 只使用X坐标
                    'b-', label='Left Wrist' if prev_idx == 0 else "", alpha=0.7)
            prev_idx = gap + 1
        if prev_idx < len(data['left_frames']):
            ax1.plot(data['left_frames'][prev_idx:], 
                    data['left_wrist'][prev_idx:, 0],  # 只使用X坐标
                    'b-', label='' if prev_idx > 0 else 'Left Wrist', alpha=0.7)
    
    if len(data['right_wrist']) > 0:
        right_gaps = np.where(np.diff(data['right_frames']) > 1)[0]
        prev_idx = 0
        for gap in right_gaps:
            # 绘制连续段 - X坐标
            ax1.plot(data['right_frames'][prev_idx:gap+1], 
                    data['right_wrist'][prev_idx:gap+1, 0],  # 只使用X坐标
                    'r-', label='Right Wrist' if prev_idx == 0 else "", alpha=0.7)
            prev_idx = gap + 1
        if prev_idx < len(data['right_frames']):
            ax1.plot(data['right_frames'][prev_idx:], 
                    data['right_wrist'][prev_idx:, 0],  # 只使用X坐标
                    'r-', label='' if prev_idx > 0 else 'Right Wrist', alpha=0.7)
    
    ax1.set_title('Wrist X-axis Movement')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X Position')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制Y轴变化
    if len(data['left_wrist']) > 0:
        left_gaps = np.where(np.diff(data['left_frames']) > 1)[0]
        prev_idx = 0
        for gap in left_gaps:
            # 绘制连续段 - Y坐标
            ax2.plot(data['left_frames'][prev_idx:gap+1], 
                    data['left_wrist'][prev_idx:gap+1, 1],  # 只使用Y坐标
                    'b-', label='Left Wrist' if prev_idx == 0 else "", alpha=0.7)
            prev_idx = gap + 1
        if prev_idx < len(data['left_frames']):
            ax2.plot(data['left_frames'][prev_idx:], 
                    data['left_wrist'][prev_idx:, 1],  # 只使用Y坐标
                    'b-', label='' if prev_idx > 0 else 'Left Wrist', alpha=0.7)
    
    if len(data['right_wrist']) > 0:
        right_gaps = np.where(np.diff(data['right_frames']) > 1)[0]
        prev_idx = 0
        for gap in right_gaps:
            # 绘制连续段 - Y坐标
            ax2.plot(data['right_frames'][prev_idx:gap+1], 
                    data['right_wrist'][prev_idx:gap+1, 1],  # 只使用Y坐标
                    'r-', label='Right Wrist' if prev_idx == 0 else "", alpha=0.7)
            prev_idx = gap + 1
        if prev_idx < len(data['right_frames']):
            ax2.plot(data['right_frames'][prev_idx:], 
                    data['right_wrist'][prev_idx:, 1],  # 只使用Y坐标
                    'r-', label='' if prev_idx > 0 else 'Right Wrist', alpha=0.7)
    
    ax2.set_title('Wrist Y-axis Movement')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y Position')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def analyze_wrist_statistics(data):
    stats = {
        'left_wrist': {},
        'right_wrist': {}
    }
    
    if len(data['left_wrist']) > 0:
        stats['left_wrist'] = {
            'mean': np.mean(data['left_wrist']),
            'std': np.std(data['left_wrist']),
            'min': np.min(data['left_wrist']),
            'max': np.max(data['left_wrist']),
            'total_frames': len(data['left_wrist'])
        }
    
    if len(data['right_wrist']) > 0:
        stats['right_wrist'] = {
            'mean': np.mean(data['right_wrist']),
            'std': np.std(data['right_wrist']),
            'min': np.min(data['right_wrist']),
            'max': np.max(data['right_wrist']),
            'total_frames': len(data['right_wrist'])
        }
    
    return stats

def save_trajectory_data(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存左手数据
    if len(data['left_wrist']) > 0:
        left_data = np.column_stack((data['left_frames'], data['left_wrist']))
        np.savetxt(
            os.path.join(output_dir, 'left_wrist_trajectory.csv'),
            left_data,
            delimiter=',',
            header='frame,value',
            comments=''
        )
    
    # 保存右手数据
    if len(data['right_wrist']) > 0:
        right_data = np.column_stack((data['right_frames'], data['right_wrist']))
        np.savetxt(
            os.path.join(output_dir, 'right_wrist_trajectory.csv'),
            right_data,
            delimiter=',',
            header='frame,value',
            comments=''
        )

def main():
    directory = "/mnt/slurm_home/htli/program/Data/hand_to_table/hamer_output_001/Hand_to_table_1_3/keypoints/"
    
    data = analyze_wrist_movement(directory)
    
    output_dir = os.path.join(os.path.dirname(directory), "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plot_wrist_trajectories(data)
    fig.savefig(os.path.join(output_dir, "wrist_movement_analysis.png"), dpi=300, bbox_inches='tight')
    
    save_trajectory_data(data, output_dir)
    
    stats = analyze_wrist_statistics(data)
    
    print("\nWrist Movement Statistics:")
    for wrist in ['left_wrist', 'right_wrist']:
        if stats[wrist]:
            print(f"\n{wrist.replace('_', ' ').title()}:")
            print(f"Total Frames: {stats[wrist]['total_frames']}")
            print(f"Mean: {stats[wrist]['mean']:.2f}")
            print(f"Std: {stats[wrist]['std']:.2f}")
            print(f"Min: {stats[wrist]['min']:.2f}")
            print(f"Max: {stats[wrist]['max']:.2f}")

if __name__ == "__main__":
    main()