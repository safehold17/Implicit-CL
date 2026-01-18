"""
Nocturne 环境视频录制工具

参考 CtRL-Sim 的实现方式：
1. 使用 matplotlib 绘制每一帧并保存为 PNG 图片
2. 使用 moviepy 将图片序列合成视频
"""
import os
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


def radians_to_degrees(radians):
    """将弧度转换为角度"""
    return radians * 180.0 / math.pi


class NocturneVideoRecorder:
    """
    Nocturne 环境视频录制器
    
    使用方式：
    1. 创建录制器：recorder = NocturneVideoRecorder(output_dir)
    2. 每步调用：recorder.capture_frame(scenario, vehicles, ...)
    3. 结束时调用：recorder.save_video(video_name)
    """
    
    def __init__(
        self,
        output_dir: str,
        fps: int = 10,
        dpi: int = 100,
        delete_images: bool = True,
    ):
        """
        初始化视频录制器
        
        Args:
            output_dir: 输出目录
            fps: 视频帧率
            dpi: 图片分辨率
            delete_images: 是否在生成视频后删除中间图片
        """
        self.output_dir = output_dir
        self.fps = fps
        self.dpi = dpi
        self.delete_images = delete_images
        
        self.frame_count = 0
        self.frames_dir = None
        
        # 保存场景数据用于计算视图范围
        self.all_positions = []
        
    def start_recording(self, name: str):
        """
        开始录制新视频
        
        Args:
            name: 视频名称（不含扩展名）
        """
        self.frame_count = 0
        self.frames_dir = os.path.join(self.output_dir, name)
        self.all_positions = []
        
        # 创建帧目录
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir, exist_ok=True)
        
    def capture_frame(
        self,
        scenario,
        vehicles: List,
        roads_data: Optional[List] = None,
        highlight_vehicle_ids: Optional[List[int]] = None,
    ):
        """
        捕获当前帧
        
        Args:
            scenario: Nocturne scenario 对象
            vehicles: 车辆列表
            roads_data: 道路数据（可选，从 scenario 提取）
            highlight_vehicle_ids: 需要高亮显示的车辆 ID 列表
        """
        if self.frames_dir is None:
            raise RuntimeError("Must call start_recording() first")
        
        # 提取车辆数据
        vehicle_data = []
        for veh in vehicles:
            pos = veh.getPosition()
            heading = veh.getHeading()
            length = veh.getLength()
            width = veh.getWidth()
            
            vehicle_data.append({
                'id': veh.getID(),
                'x': pos.x,
                'y': pos.y,
                'heading': heading,
                'length': length,
                'width': width,
            })
            
            # 记录位置用于计算视图范围
            self.all_positions.append([pos.x, pos.y])
        
        # 保存帧
        self._save_frame(vehicle_data, roads_data, highlight_vehicle_ids)
        self.frame_count += 1
    
    def _save_frame(
        self,
        vehicle_data: List[Dict],
        roads_data: Optional[List],
        highlight_vehicle_ids: Optional[List[int]],
    ):
        """绘制并保存单帧"""
        plt.figure(figsize=(10, 10))
        
        # 计算视图范围
        if len(self.all_positions) > 0:
            positions = np.array(self.all_positions)
            x_min = np.min(positions[:, 0]) - 25
            x_max = np.max(positions[:, 0]) + 25
            y_min = np.min(positions[:, 1]) - 25
            y_max = np.max(positions[:, 1]) + 25
            
            # 保持正方形视图
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range > y_range:
                diff = (x_range - y_range) / 2
                y_min -= diff
                y_max += diff
            else:
                diff = (y_range - x_range) / 2
                x_min -= diff
                x_max += diff
        else:
            x_min, x_max = -50, 50
            y_min, y_max = -50, 50
        
        # 绘制道路（如果提供）
        if roads_data is not None:
            self._draw_roads(roads_data)
        
        # 绘制车辆
        for veh in vehicle_data:
            is_highlight = (
                highlight_vehicle_ids is not None 
                and veh['id'] in highlight_vehicle_ids
            )
            self._draw_vehicle(veh, is_highlight)
        
        # 设置视图
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', adjustable='box')
        
        # 隐藏坐标轴
        plt.tick_params(
            left=False, right=False, labelleft=False,
            labelbottom=False, bottom=False
        )
        
        # 保存图片
        plt.tight_layout()
        frame_path = os.path.join(self.frames_dir, f'frame_{self.frame_count:04d}.png')
        plt.savefig(frame_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _draw_roads(self, roads_data: List):
        """绘制道路网络"""
        for road in roads_data:
            geometry = road.get('geometry', [])
            road_type = road.get('type', 'unknown')
            
            if isinstance(geometry, dict):
                # 单点（如停止标志）
                plt.scatter(
                    geometry['x'], geometry['y'],
                    color='red', s=20, zorder=1
                )
            elif isinstance(geometry, list) and len(geometry) > 0:
                # 线段
                x_coords = [p['x'] for p in geometry]
                y_coords = [p['y'] for p in geometry]
                
                # 根据道路类型设置颜色
                if road_type == 'road_edge':
                    color = 'grey'
                    linewidth = 0.5
                else:
                    color = 'lightgray'
                    linewidth = 0.3
                
                plt.plot(x_coords, y_coords, color=color, linewidth=linewidth, zorder=1)
    
    def _draw_vehicle(self, veh: Dict, is_highlight: bool = False):
        """绘制单个车辆"""
        x, y = veh['x'], veh['y']
        heading = veh['heading']
        length = veh['length'] * 0.8
        width = veh['width'] * 0.8
        
        # 设置颜色
        if is_highlight:
            color = '#ff6b6b'  # 红色高亮
            alpha = 0.8
        else:
            color = '#ffde8b'  # 默认黄色
            alpha = 0.5
        
        # 绘制车辆边界框
        bbox_x_min = x - width / 2
        bbox_y_min = y - length / 2
        
        rectangle = mpatches.FancyBboxPatch(
            (bbox_x_min, bbox_y_min),
            width, length,
            ec='black', fc=color,
            linewidth=0.35,
            alpha=alpha,
            boxstyle=mpatches.BoxStyle("Round", pad=0.3),
            zorder=4
        )
        
        # 应用旋转
        tr = transforms.Affine2D().rotate_deg_around(
            x, y, radians_to_degrees(heading) - 90
        ) + plt.gca().transData
        rectangle.set_transform(tr)
        
        plt.gca().add_patch(rectangle)
        
        # 绘制朝向箭头
        heading_length = length / 2 + 1.5
        line_end_x = x + heading_length * math.cos(heading)
        line_end_y = y + heading_length * math.sin(heading)
        
        plt.plot(
            [x, line_end_x], [y, line_end_y],
            color='black', zorder=6,
            alpha=0.25, linewidth=0.25
        )
    
    def save_video(self, video_name: str) -> str:
        """
        将帧序列合成为视频
        
        Args:
            video_name: 视频文件名（不含扩展名）
        
        Returns:
            视频文件路径
        """
        if self.frames_dir is None or self.frame_count == 0:
            raise RuntimeError("No frames to save")
        
        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            raise ImportError(
                "moviepy is required for video generation. "
                "Install it with: pip install moviepy"
            )
        
        # 获取所有图片文件
        images = [
            os.path.join(self.frames_dir, f'frame_{i:04d}.png')
            for i in range(self.frame_count)
        ]
        
        # 检查文件是否存在
        images = [img for img in images if os.path.exists(img)]
        
        if len(images) == 0:
            raise RuntimeError("No image files found")
        
        # 创建视频
        video_path = os.path.join(self.output_dir, f'{video_name}.mp4')
        clip = ImageSequenceClip(images, fps=self.fps)
        clip.write_videofile(video_path, codec='libx264', logger=None)
        
        # 删除中间图片
        if self.delete_images:
            shutil.rmtree(self.frames_dir)
        
        self.frames_dir = None
        self.frame_count = 0
        self.all_positions = []
        
        return video_path
    
    def close(self):
        """清理资源"""
        if self.frames_dir is not None and self.delete_images:
            if os.path.exists(self.frames_dir):
                shutil.rmtree(self.frames_dir)
        
        self.frames_dir = None
        self.frame_count = 0
        self.all_positions = []


def create_video_from_episode(
    output_dir: str,
    video_name: str,
    scenario,
    episode_data: List[Dict],
    fps: int = 10,
    dpi: int = 100,
    delete_images: bool = True,
) -> str:
    """
    从完整 episode 数据创建视频（事后生成）
    
    Args:
        output_dir: 输出目录
        video_name: 视频文件名
        scenario: Nocturne scenario 对象
        episode_data: Episode 数据列表，每个元素包含一帧的信息
        fps: 帧率
        dpi: 分辨率
        delete_images: 是否删除中间图片
    
    Returns:
        视频文件路径
    """
    recorder = NocturneVideoRecorder(output_dir, fps, dpi, delete_images)
    recorder.start_recording(video_name)
    
    for frame_data in tqdm(episode_data, desc="Generating frames"):
        vehicles = frame_data.get('vehicles', [])
        roads_data = frame_data.get('roads_data')
        highlight_ids = frame_data.get('highlight_vehicle_ids')
        
        recorder.capture_frame(scenario, vehicles, roads_data, highlight_ids)
    
    video_path = recorder.save_video(video_name)
    recorder.close()
    
    return video_path
