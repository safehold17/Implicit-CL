"""
Visualization helpers for Nocturne CtrlSim environment.
"""
import math
from typing import Optional

import numpy as np

from .video_recorder import NocturneVideoRecorder


class VisualizationMixin:
    def render(self, mode='human'):
        """Render environment (static screenshot)"""
        if mode not in ['human', 'rgb_array', 'level']: # render is the gym standard parameter
            raise NotImplementedError

        if self.scenario is None or not self.vehicles:
            return None

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import matplotlib.patches as mpatches
        import matplotlib.transforms as transforms

        vehicle_data = []
        positions = []
        for veh in self.vehicles:
            pos = veh.getPosition()
            if pos.x == -10000 and pos.y == -10000:
                continue
            vehicle_data.append({
                'id': veh.getID(),
                'x': pos.x,
                'y': pos.y,
                'heading': veh.getHeading(),
                'length': veh.getLength(),
                'width': veh.getWidth(),
            })
            positions.append([pos.x, pos.y])

        if not vehicle_data:
            return None

        fig = Figure(figsize=(10, 10), dpi=200)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        def _draw_road(geometry, color, linewidth):
            if isinstance(geometry, dict):
                ax.scatter(geometry['x'], geometry['y'], color='red', s=20, zorder=1)
            elif isinstance(geometry, list) and len(geometry) > 0:
                xs = [p['x'] for p in geometry]
                ys = [p['y'] for p in geometry]
                ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=1)

        roads_data = self._road_graph_cache
        if roads_data is None and self.scenario is not None:
            roads_data = self.data_bridge.get_road_data(self.scenario)

        if roads_data:
            for road in roads_data:
                if road.get('type') == 'road_edge':
                    _draw_road(road.get('geometry', []), color='grey', linewidth=0.5)
            for road in roads_data:
                if road.get('type') != 'road_edge':
                    _draw_road(road.get('geometry', []), color='lightgray', linewidth=0.3)

        positions = np.array(positions)
        x_min = np.min(positions[:, 0]) - 25
        x_max = np.max(positions[:, 0]) + 25
        y_min = np.min(positions[:, 1]) - 25
        y_max = np.max(positions[:, 1]) + 25

        if (x_max - x_min) > (y_max - y_min):
            diff = (x_max - x_min) - (y_max - y_min)
            y_min -= diff / 2
            y_max += diff / 2
        else:
            diff = (y_max - y_min) - (x_max - x_min)
            x_min -= diff / 2
            x_max += diff / 2

        line_scale = (x_max - x_min) / 140 if x_max > x_min else 1.0
        lw = 0.35 / line_scale
        heading_lw = 0.25 / line_scale

        highlight_ids = set()
        if self.ego_vehicle is not None:
            highlight_ids.add(self.ego_vehicle.getID())
        opponent_ids = set(self.opponent_vehicle_ids) if self.opponent_vehicle_ids else set()
        show_tilting_params = getattr(self, 'show_tilting_params', True)
        show_vehicle_ids = getattr(self, 'show_vehicle_ids', True)
        show_ego_vehicle_selection = getattr(self, 'show_ego_vehicle_selection', True)
        tilt_by_vehicle_id = {}
        if show_tilting_params and self.current_level is not None and opponent_ids:
            if self.tilting_mode == 'global':
                tilt_tuple = (
                    self.current_level.goal_tilt,
                    self.current_level.veh_veh_tilt,
                    self.current_level.veh_edge_tilt,
                )
                for veh_id in opponent_ids:
                    tilt_by_vehicle_id[veh_id] = tilt_tuple
            else:
                per = self.current_level.per_vehicle_tilting
                if per:
                    sorted_opponent_ids = sorted(self.opponent_vehicle_ids)
                    for i, veh_id in enumerate(sorted_opponent_ids):
                        base = 3 * i
                        if base + 2 < len(per):
                            tilt_by_vehicle_id[veh_id] = (per[base], per[base + 1], per[base + 2])

        vehicle_patches = {}
        tilt_text_specs = []
        for veh in vehicle_data:
            is_highlight = veh['id'] in highlight_ids
            is_opponent = (not is_highlight) and veh['id'] in opponent_ids
            if is_highlight:
                color = '#ff6b6b'
                alpha = 0.8
            elif is_opponent:
                color = '#4aa3ff'
                alpha = 0.8
            else:
                color = '#ffde8b'
                alpha = 0.5

            length = veh['length'] * 0.8
            width = veh['width'] * 0.8
            bbox_x_min = veh['x'] - width / 2
            bbox_y_min = veh['y'] - length / 2

            rectangle = mpatches.FancyBboxPatch(
                (bbox_x_min, bbox_y_min),
                width, length,
                ec='black', fc=color, linewidth=lw, alpha=alpha,
                boxstyle=mpatches.BoxStyle("Round", pad=0.3),
                zorder=4
            )

            tr = transforms.Affine2D().rotate_deg_around(
                veh['x'], veh['y'], math.degrees(veh['heading']) - 90
            ) + ax.transData
            rectangle.set_transform(tr)
            ax.add_patch(rectangle)
            if is_highlight or is_opponent:
                vehicle_patches[veh['id']] = rectangle
            if show_vehicle_ids and (is_highlight or is_opponent):
                ax.text(
                    veh['x'],
                    veh['y'],
                    f"{veh['id']}",
                    fontsize=5,
                    color='black',
                    ha='center',
                    va='center',
                    zorder=7,
                )

            heading_length = length / 2 + 1.5
            line_end_x = veh['x'] + heading_length * math.cos(veh['heading'])
            line_end_y = veh['y'] + heading_length * math.sin(veh['heading'])
            ax.plot(
                [veh['x'], line_end_x], [veh['y'], line_end_y],
                color='black', zorder=6, alpha=0.25, linewidth=heading_lw
            )
            if show_tilting_params and is_opponent and veh['id'] in tilt_by_vehicle_id:
                tilt_vals = tilt_by_vehicle_id[veh['id']]
                is_horizontal = abs(math.cos(veh['heading'])) >= abs(math.sin(veh['heading']))
                if is_horizontal:
                    text_x = veh['x']
                    text_y = veh['y'] + width / 2 + width * 0.6
                    ha, va = 'center', 'bottom'
                else:
                    text_x = veh['x'] - width / 2 - width * 0.6
                    text_y = veh['y']
                    ha, va = 'right', 'center'
                text_artist = ax.text(
                    text_x,
                    text_y,
                    f"[{tilt_vals[0]}, {tilt_vals[1]}, {tilt_vals[2]}]",
                    fontsize=6,
                    color='black',
                    ha=ha,
                    va=va,
                    zorder=7,
                )
                tilt_text_specs.append({
                    'veh_id': veh['id'],
                    'veh': veh,
                    'width': width,
                    'is_horizontal': is_horizontal,
                    'text_artist': text_artist,
                })

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        if self.current_level is not None:
            ax.text(
                0.01,
                0.99,
                f"scenario: {self.current_level.scenario_id}",
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=8,
                color='black',
                zorder=8,
            )
            if show_ego_vehicle_selection:
                selection_mode = getattr(self, 'ego_selection_mode', 'unknown')
                if selection_mode == 'interesting':
                    selection_text = 'interesting vehicle'
                elif selection_mode == 'dense':
                    selection_text = 'dense vehicle'
                else:
                    selection_text = 'unknown'
                ax.text(
                    0.01,
                    0.965,
                    selection_text,
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    fontsize=8,
                    color='black',
                    zorder=8,
                )

        def _bboxes_overlap(bbox_a, bbox_b):
            return (
                bbox_a.x0 <= bbox_b.x1 and bbox_a.x1 >= bbox_b.x0
                and bbox_a.y0 <= bbox_b.y1 and bbox_a.y1 >= bbox_b.y0
            )

        fig.tight_layout()
        canvas.draw()
        renderer = canvas.get_renderer()
        if vehicle_patches and tilt_text_specs:
            vehicle_bbox_by_id = {
                veh_id: patch.get_window_extent(renderer)
                for veh_id, patch in vehicle_patches.items()
            }
            for spec in tilt_text_specs:
                veh_id = spec['veh_id']
                text_artist = spec['text_artist']
                text_bbox = text_artist.get_window_extent(renderer)
                overlap = any(
                    _bboxes_overlap(text_bbox, bbox)
                    for other_id, bbox in vehicle_bbox_by_id.items()
                    if other_id != veh_id
                )
                if overlap:
                    veh = spec['veh']
                    width = spec['width']
                    if spec['is_horizontal']:
                        text_artist.set_position((veh['x'], veh['y'] - width / 2 - width * 0.6))
                        text_artist.set_va('top')
                    else:
                        text_artist.set_position((veh['x'] + width / 2 + width * 0.6, veh['y']))
                        text_artist.set_ha('left')
            canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        fig.clear()

        return image

    # TODO: test recording function, fps / dpi no need to be set
    def start_recording(self, output_dir: str, video_name: str, fps: int = 10, dpi: int = 100):
        """    
        Args:
            output_dir: Output directory
            video_name: Video file name (without extension)
            fps: Frame rate
            dpi: Resolution
        """
        if self.video_recorder is None:
            self.video_recorder = NocturneVideoRecorder(
                output_dir=output_dir,
                fps=fps,
                dpi=dpi,
                delete_images=True
            )
        
        self.video_recorder.start_recording(video_name)
        self.recording_video = True
        
        # Capture first frame (initial state)
        if self.scenario is not None and self.vehicles:
            self.video_recorder.capture_frame(
                self.scenario,
                self.vehicles,
                highlight_vehicle_ids=[self.ego_vehicle.getID()] if self.ego_vehicle else None
            )
    
    def stop_recording(self, video_name: Optional[str] = None) -> Optional[str]:
        """
        Args:
            video_name: Video file name (if different from start_recording)
        
        Returns:
            Video file path, None if not recording
        """
        if not self.recording_video or self.video_recorder is None:
            return None
        
        self.recording_video = False
        
        try:
            if video_name is None:
                # Use default name
                if self.current_level:
                    video_name = f"scenario_{self.current_level.scenario_id}"
                else:
                    video_name = "episode"
            
            video_path = self.video_recorder.save_video(video_name)
            return video_path
        except Exception as e:
            print(f"Error saving video: {e}")
            return None
