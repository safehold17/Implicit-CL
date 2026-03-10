"""
Nocturne 环境视频录制工具
Nocturne environment video recorder.

参考 CtRL-Sim 的实现方式：
Implementation follows the CtRL-Sim approach:
1. 使用 matplotlib 绘制每一帧并保存为 PNG 图片
1. Use matplotlib to render each frame and save it as a PNG image.
2. 使用 moviepy 将图片序列合成视频
2. Use moviepy to compose the image sequence into a video.
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

from .common import (
    compute_square_view_bounds,
    is_valid_world_position,
    radians_to_degrees,
)


class NocturneVideoRecorder:
    """
    Nocturne 环境视频录制器
    Nocturne environment video recorder.

    使用方式：
    Usage:
    1. 创建录制器：recorder = NocturneVideoRecorder(output_dir)
    1. Create the recorder: recorder = NocturneVideoRecorder(output_dir).
    2. 每步调用：recorder.capture_frame(scenario, vehicles, ...)
    2. Call recorder.capture_frame(scenario, vehicles, ...) at each step.
    3. 结束时调用：recorder.save_video(video_name)
    3. Call recorder.save_video(video_name) at the end.
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
        Initialize the video recorder.

        Args:
        output_dir: 输出目录
        output_dir: output directory.
        fps: 视频帧率
        fps: video frame rate.
        dpi: 图片分辨率
        dpi: image resolution.
        delete_images: 是否在生成视频后删除中间图片
        delete_images: whether to delete intermediate images after the video is created.
        """
        self.output_dir = output_dir
        self.fps = fps
        self.dpi = dpi
        self.delete_images = delete_images
        
        self.frame_count = 0
        self.frames_dir = None

        # 保存场景数据用于计算视图范围
        # Store scenario data to compute view bounds.
        self.all_positions = []
        self._debug_invalid_logged = False
        
    def start_recording(self, name: str):
        """
        开始录制新视频
        Start recording a new video.

        Args:
        name: 视频名称（不含扩展名）
        name: video name without extension.
        """
        self.frame_count = 0
        self.frames_dir = os.path.join(self.output_dir, name)
        self.all_positions = []
        self._debug_invalid_logged = False
        # Always write debug log to videos/nocturne_video_debug.log (user request)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建帧目录
        # Create the frame directory.
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir, exist_ok=True)
        
    def capture_frame(
        self,
        scenario,
        vehicles: List,
        roads_data: Optional[List] = None,
        highlight_vehicle_ids: Optional[List[int]] = None,
        opponent_vehicle_ids: Optional[List[int]] = None,
        goal_points_by_id: Optional[Dict[int, np.ndarray]] = None,
        scenario_id: Optional[str] = None,
        show_vehicle_ids: bool = False,
    ):
        """
        捕获当前帧
        Capture the current frame.

        Args:
        scenario: Nocturne scenario 对象
        scenario: Nocturne scenario object.
        vehicles: 车辆列表
        vehicles: list of vehicles.
        roads_data: 道路数据（可选，从 scenario 提取）
        roads_data: road data, optionally extracted from the scenario.
        highlight_vehicle_ids: 需要高亮显示的车辆 ID 列表
        highlight_vehicle_ids: list of vehicle IDs to highlight.
        """
        if self.frames_dir is None:
            raise RuntimeError("Must call start_recording() first")
        
        # 提取车辆数据
        # Extract vehicle data.
        vehicle_data = []
        for veh in vehicles:
            pos = veh.getPosition()
            heading = veh.getHeading()
            length = veh.getLength()
            width = veh.getWidth()

            x = pos.x
            y = pos.y

            if not is_valid_world_position(x, y):
                continue

            vehicle_data.append({
                'id': veh.getID(),
                'x': x,
                'y': y,
                'heading': heading,
                'length': length,
                'width': width,
            })
            
            # 记录位置用于计算视图范围
            # Record positions to compute view bounds.
            self.all_positions.append([x, y])

        # 保存帧
        # Save the frame.
        self._save_frame(
            vehicle_data,
            roads_data,
            highlight_vehicle_ids,
            opponent_vehicle_ids,
            goal_points_by_id,
            scenario_id,
            show_vehicle_ids,
        )
        self.frame_count += 1
    
    def _save_frame(
        self,
        vehicle_data: List[Dict],
        roads_data: Optional[List],
        highlight_vehicle_ids: Optional[List[int]],
        opponent_vehicle_ids: Optional[List[int]],
        goal_points_by_id: Optional[Dict[int, np.ndarray]],
        scenario_id: Optional[str],
        show_vehicle_ids: bool,
    ):
        """
        绘制并保存单帧
        Draw and save a single frame.
        """
        plt.figure(figsize=(10, 10))
        
        # 计算视图范围
        # Compute the view bounds.
        x_min, x_max, y_min, y_max = compute_square_view_bounds(self.all_positions)
        
        # 绘制道路（如果提供）
        # Draw roads when they are provided.
        if roads_data is not None:
            self._draw_roads(roads_data)
        
        # 绘制车辆
        # Draw vehicles.
        highlight_ids = set(highlight_vehicle_ids) if highlight_vehicle_ids else set()
        opponent_ids = set(opponent_vehicle_ids) if opponent_vehicle_ids else set()
        for veh in vehicle_data:
            is_highlight = (
                highlight_ids
                and veh['id'] in highlight_ids
            )
            is_opponent = (not is_highlight) and (veh['id'] in opponent_ids)
            self._draw_vehicle(veh, is_highlight, is_opponent)
            if show_vehicle_ids and (is_highlight or is_opponent):
                plt.gca().text(
                    veh['x'],
                    veh['y'],
                    f"{veh['id']}",
                    fontsize=5,
                    color='black',
                    ha='center',
                    va='center',
                    zorder=7,
                )

        if goal_points_by_id:
            for veh_id, goal_pos in goal_points_by_id.items():
                if goal_pos is None or len(goal_pos) < 2:
                    continue
                x = float(goal_pos[0])
                y = float(goal_pos[1])
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                is_highlight = veh_id in highlight_ids
                is_opponent = (not is_highlight) and (veh_id in opponent_ids)
                self._draw_goal_point(x, y, is_highlight, is_opponent)
        
        # 设置视图
        # Set the view.
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', adjustable='box')
        
        # 隐藏坐标轴
        # Hide axes.
        plt.tick_params(
            left=False, right=False, labelleft=False,
            labelbottom=False, bottom=False
        )

        if scenario_id is not None:
            ax = plt.gca()
            ax.text(
                0.01,
                0.99,
                f"scenario: {scenario_id}",
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=8,
                color='black',
                zorder=8,
            )
        
        # 调整边距以确保内容显示完整，但保持固定尺寸
        # Adjust margins to keep the content fully visible while preserving a fixed size.
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # 保存图片
        # Save the image.
        frame_path = os.path.join(self.frames_dir, f'frame_{self.frame_count:04d}.png')
        plt.savefig(frame_path, dpi=self.dpi)
        plt.close()
    
    def _draw_roads(self, roads_data: List):
        """
        绘制道路网络
        Draw the road network.
        """
        for road in roads_data:
            geometry = road.get('geometry', [])
            road_type = road.get('type', 'unknown')
            
            if isinstance(geometry, dict):
                # 单点（如停止标志）
                # Single point, such as a stop sign.
                plt.scatter(
                    geometry['x'], geometry['y'],
                    color='red', s=20, zorder=1
                )
            elif isinstance(geometry, list) and len(geometry) > 0:
                # 线段
                # Line segment.
                x_coords = [p['x'] for p in geometry]
                y_coords = [p['y'] for p in geometry]
                
                # 根据道路类型设置颜色
                # Set colors according to road type.
                if road_type == 'road_edge':
                    color = 'grey'
                    linewidth = 0.5
                else:
                    color = 'lightgray'
                    linewidth = 0.3
                
                plt.plot(x_coords, y_coords, color=color, linewidth=linewidth, zorder=1)
    
    def _draw_vehicle(
        self,
        veh: Dict,
        is_highlight: bool = False,
        is_opponent: bool = False,
    ):
        """
        绘制单个车辆
        Draw a single vehicle.
        """
        x, y = veh['x'], veh['y']
        heading = veh['heading']
        length = veh['length'] * 0.8
        width = veh['width'] * 0.8
        
        # 设置颜色
        # Set the color.
        if is_highlight:
            # 红色高亮
            # Highlighted red.
            color = '#ff6b6b'
            alpha = 0.8
        elif is_opponent:
            # 对手车辆蓝色
            # Blue for opponent vehicles.
            color = '#4aa3ff'
            alpha = 0.8
        else:
            # 默认黄色
            # Default yellow.
            color = '#ffde8b'
            alpha = 0.5
        
        # 绘制车辆边界框
        # Draw the vehicle bounding box.
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
        # Apply rotation.
        tr = transforms.Affine2D().rotate_deg_around(
            x, y, radians_to_degrees(heading) - 90
        ) + plt.gca().transData
        rectangle.set_transform(tr)
        
        plt.gca().add_patch(rectangle)
        
        # 绘制朝向箭头
        # Draw the heading arrow.
        heading_length = length / 2 + 1.5
        line_end_x = x + heading_length * math.cos(heading)
        line_end_y = y + heading_length * math.sin(heading)
        
        plt.plot(
            [x, line_end_x], [y, line_end_y],
            color='black', zorder=6,
            alpha=0.25, linewidth=0.25
        )

    def _draw_goal_point(
        self,
        x: float,
        y: float,
        is_highlight: bool = False,
        is_opponent: bool = False,
    ):
        if is_highlight:
            color = '#ff6b6b'
            alpha = 0.8
        elif is_opponent:
            color = '#4aa3ff'
            alpha = 0.8
        else:
            color = '#ffde8b'
            alpha = 0.5

        inner_radius = 0.6
        outer_radius = 1.6

        inner = mpatches.Circle(
            (x, y),
            radius=inner_radius,
            ec='none',
            fc=color,
            alpha=alpha,
            zorder=5,
        )
        outer = mpatches.Circle(
            (x, y),
            radius=outer_radius,
            fill=False,
            ec=color,
            linewidth=0.35,
            linestyle=(0, (2, 2)),
            alpha=alpha,
            zorder=5,
        )
        plt.gca().add_patch(inner)
        plt.gca().add_patch(outer)
    
    def save_video(self, video_name: str) -> str:
        """
        将帧序列合成为视频
        Compose the frame sequence into a video.

        Args:
        video_name: 视频文件名（不含扩展名）
        video_name: video filename without extension.

        Returns:
        视频文件路径
        Path to the video file.
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
        # Get all image files.
        images = [
            os.path.join(self.frames_dir, f'frame_{i:04d}.png')
            for i in range(self.frame_count)
        ]
        
        # 检查文件是否存在
        # Check whether the file exists.
        images = [img for img in images if os.path.exists(img)]
        
        if len(images) == 0:
            raise RuntimeError("No image files found")
        
        # 创建视频
        # Create the video.
        video_path = os.path.join(self.output_dir, f'{video_name}.mp4')
        clip = ImageSequenceClip(images, fps=self.fps)
        clip.write_videofile(video_path, codec='libx264', logger=None)
        
        # 删除中间图片
        # Delete intermediate images.
        if self.delete_images:
            shutil.rmtree(self.frames_dir)
        
        self.frames_dir = None
        self.frame_count = 0
        self.all_positions = []
        
        return video_path
    
    def close(self):
        """
        清理资源
        Clean up resources.
        """
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
    scenario_id: Optional[str] = None,
) -> str:
    """
    从完整 episode 数据创建视频（事后生成）
    Create a video from full episode data after the fact.

    Args:
    output_dir: 输出目录
    output_dir: output directory.
    video_name: 视频文件名
    video_name: video filename.
    scenario: Nocturne scenario 对象
    scenario: Nocturne scenario object.
    episode_data: Episode 数据列表，每个元素包含一帧的信息
    episode_data: episode-data list, where each element contains one frame of information.
    fps: 帧率
    fps: frame rate.
    dpi: 分辨率
    dpi: resolution.
    delete_images: 是否删除中间图片
    delete_images: whether to delete intermediate images.
    scenario_id: 场景 ID（用于帧标注）
    scenario_id: scenario ID used for frame annotation.

    Returns:
    视频文件路径
    Path to the video file.
    """
    recorder = NocturneVideoRecorder(output_dir, fps, dpi, delete_images)
    recorder.start_recording(video_name)
    
    for frame_data in tqdm(episode_data, desc="Generating frames"):
        vehicles = frame_data.get('vehicles', [])
        roads_data = frame_data.get('roads_data')
        highlight_ids = frame_data.get('highlight_vehicle_ids')
        opponent_ids = frame_data.get('opponent_vehicle_ids')

        recorder.capture_frame(
            scenario,
            vehicles,
            roads_data,
            highlight_ids,
            opponent_ids,
            scenario_id=scenario_id,
        )
    
    video_path = recorder.save_video(video_name)
    recorder.close()
    
    return video_path
