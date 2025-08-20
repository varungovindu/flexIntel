#!/usr/bin/env python3
"""
Pushup Tracker - flexIntel Exercise Module
Professional pushup form tracking with real-time feedback
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, cast

import cv2
import mediapipe as mp
import numpy as np




# -------------------------------
# Configuration dataclasses
# -------------------------------
@dataclass
class CounterConfig:
    top_angle_deg: float = 170.0         # Angle when arms are extended (up position)
    bottom_angle_deg: float = 90.0       # Angle when arms are bent (down position)
    hold_frames_required: int = 3         # Frames to hold top/bottom to register state
    smoothing_alpha: float = 0.2          # EMA smoothing factor for elbow angle
    min_visibility: float = 0.6           # Min landmark visibility to be considered valid
    body_alignment_threshold: float = 0.1 # Threshold for body alignment check


@dataclass
class DrawConfig:
    show_pose_skeleton: bool = True
    mirror_view: bool = True
    show_debug: bool = False


# -------------------------------
# Pose + utility helpers
# -------------------------------
mp_pose: Any = mp.solutions.pose
mp_drawing: Any = mp.solutions.drawing_utils
cv2 = cast(Any, cv2)

# Landmark indices for pushup tracking
DICT_FEATURES = {
    'left': {
        'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        'elbow': mp_pose.PoseLandmark.LEFT_ELBOW.value,
        'wrist': mp_pose.PoseLandmark.LEFT_WRIST.value,
        'hip': mp_pose.PoseLandmark.LEFT_HIP.value,
        'ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value,
    },
    'right': {
        'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        'wrist': mp_pose.PoseLandmark.RIGHT_WRIST.value,
        'hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
        'ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    }, 
}

def open_video_capture(input_source, width: int, height: int, fps: int):
    backend_preferences = []
    if isinstance(input_source, int):
        backend_preferences = [getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_MSMF", 1400), getattr(cv2, "CAP_ANY", 0)]
    else:
        backend_preferences = [getattr(cv2, "CAP_ANY", 0)]

    last_cap = None
    for backend in backend_preferences:
        try:
            cap = cv2.VideoCapture(input_source, backend)
        except Exception:
            cap = cv2.VideoCapture(input_source)
        if width and width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height and height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps and fps > 0:
            cap.set(cv2.CAP_PROP_FPS, fps)
        if cap.isOpened():
            return cap
        last_cap = cap
        try:
            cap.release()
        except Exception:
            pass
    return last_cap if last_cap and last_cap.isOpened() else last_cap


def get_landmark_xy(landmark, frame_w: int, frame_h: int) -> Tuple[int, int]:
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def compute_angle_degrees(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> Optional[float]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    ab = np.array([ax - bx, ay - by], dtype=np.float32)
    cb = np.array([cx - bx, cy - by], dtype=np.float32)
    ab_norm = np.linalg.norm(ab)
    cb_norm = np.linalg.norm(cb)
    if ab_norm < 1e-6 or cb_norm < 1e-6:
        return None
    cos_angle = float(np.dot(ab, cb) / (ab_norm * cb_norm))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def check_body_alignment(shoulder_xy: Tuple[int, int], hip_xy: Tuple[int, int], 
                        ankle_xy: Tuple[int, int]) -> bool:
    """Check if body is in a straight line for proper pushup form"""
    # Calculate the angle between shoulder-hip-ankle line
    angle = compute_angle_degrees(shoulder_xy, hip_xy, ankle_xy)
    if angle is None:
        return False
    
    # Body should be roughly straight (angle close to 180 degrees)
    return abs(angle - 180.0) < 15.0


# -------------------------------
# Rep counting state machine
# -------------------------------
@dataclass
class SideState:
    side_name: str
    rep_count: int = 0
    direction_up: bool = False  # False = going down, True = going up
    ema_angle: Optional[float] = None
    top_hold_frames: int = 0
    bottom_hold_frames: int = 0
    body_aligned: bool = False
    last_feedback: str = ""
    last_color: Tuple[int, int, int] = (255, 255, 255)
    progress_01: float = 0.0

    def update(self,
               angle_raw: Optional[float],
               shoulder_xy: Tuple[int, int],
               elbow_xy: Tuple[int, int],
               wrist_xy: Tuple[int, int],
               hip_xy: Tuple[int, int],
               ankle_xy: Tuple[int, int],
               frame_h: int,
               cfg: CounterConfig) -> None:
        # Validate angle
        if angle_raw is None:
            self.last_feedback = "Landmarks unstable"
            self.last_color = (0, 165, 255)  # orange
            return

        # Check body alignment
        self.body_aligned = check_body_alignment(shoulder_xy, hip_xy, ankle_xy)

        # Apply EMA smoothing
        if self.ema_angle is None:
            self.ema_angle = angle_raw
        else:
            self.ema_angle = cfg.smoothing_alpha * angle_raw + (1 - cfg.smoothing_alpha) * self.ema_angle

        angle = self.ema_angle

        # Update state machine
        if not self.direction_up:  # Going down
            if angle <= cfg.bottom_angle_deg:
                self.bottom_hold_frames += 1
                if self.bottom_hold_frames >= cfg.hold_frames_required:
                    self.direction_up = True
                    self.bottom_hold_frames = 0
                    self.top_hold_frames = 0
                    self.last_feedback = "Go down more"
                    self.last_color = (0, 255, 255)  # cyan
            else:
                self.bottom_hold_frames = 0
                self.last_feedback = "Lower your body"
                self.last_color = (0, 255, 255)  # cyan
        else:  # Going up
            if angle >= cfg.top_angle_deg:
                self.top_hold_frames += 1
                if self.top_hold_frames >= cfg.hold_frames_required:
                    self.direction_up = False
                    self.top_hold_frames = 0
                    self.bottom_hold_frames = 0
                    self.rep_count += 1
                    self.last_feedback = "Great! Push up!"
                    self.last_color = (0, 255, 0)  # green
            else:
                self.top_hold_frames = 0
                self.last_feedback = "Push up!"
                self.last_color = (0, 255, 0)  # green

        # Calculate progress (0.0 to 1.0)
        if self.direction_up:
            # Going from bottom to top
            progress = (angle - cfg.bottom_angle_deg) / (cfg.top_angle_deg - cfg.bottom_angle_deg)
        else:
            # Going from top to bottom
            progress = (cfg.top_angle_deg - angle) / (cfg.top_angle_deg - cfg.bottom_angle_deg)
        
        self.progress_01 = np.clip(progress, 0.0, 1.0)

        # Form feedback
        if not self.body_aligned:
            self.last_feedback = "Keep body straight!"
            self.last_color = (0, 165, 255)  # orange


# -------------------------------
# Professional overlay drawer
# -------------------------------
class OverlayDrawer:
    def __init__(self, cfg: DrawConfig):
        self.cfg = cfg

    def draw_rep_counter(self, frame: np.ndarray, states: Dict[str, SideState], frame_w: int, frame_h: int):
        """Draw rep counter with professional styling"""
        # Background panel
        panel_width = 300
        panel_height = 120
        panel_x = frame_w - panel_width - 20
        panel_y = 80
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "PUSHUP COUNTER", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Rep counts
        y_offset = 50
        for side, state in states.items():
            side_name = side.capitalize()
            rep_text = f"{side_name}: {state.rep_count}"
            cv2.putText(frame, rep_text, (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
        
        # Total reps
        total_reps = sum(state.rep_count for state in states.values())
        total_text = f"TOTAL: {total_reps}"
        cv2.putText(frame, total_text, (panel_x + 10, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    def draw_form_feedback(self, frame: np.ndarray, states: Dict[str, SideState], frame_w: int, frame_h: int):
        """Draw form feedback with professional styling"""
        # Find the most relevant feedback
        feedbacks = [state.last_feedback for state in states.values() if state.last_feedback]
        if not feedbacks:
            return
        
        # Use the most recent feedback
        feedback = feedbacks[-1]
        
        # Background panel
        panel_width = 400
        panel_height = 60
        panel_x = 20
        panel_y = frame_h - panel_height - 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (100, 100, 100), 2)
        
        # Feedback text
        cv2.putText(frame, feedback, (panel_x + 10, panel_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_progress_bars(self, frame: np.ndarray, states: Dict[str, SideState], frame_w: int, frame_h: int):
        """Draw progress bars for each side"""
        bar_width = 200
        bar_height = 15
        bar_x = 20
        bar_y = 80
        
        for i, (side, state) in enumerate(states.items()):
            y_pos = bar_y + i * (bar_height + 10)
            
            # Background bar
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), (100, 100, 100), -1)
            
            # Progress bar
            progress_width = int(bar_width * state.progress_01)
            if progress_width > 0:
                cv2.rectangle(frame, (bar_x, y_pos), (bar_x + progress_width, y_pos + bar_height), state.last_color, -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), (255, 255, 255), 1)
            
            # Label
            label = f"{side.capitalize()}: {state.progress_01:.1%}"
            cv2.putText(frame, label, (bar_x + bar_width + 10, y_pos + bar_height - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# -------------------------------
# Main application
# -------------------------------
def run_app(args):
    """Main application loop"""
    # Initialize pose detection
    pose = mp_pose.Pose(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    # Initialize video capture
    cap = open_video_capture(args.input, args.width, args.height, args.fps)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        return

    # Set capture properties
    # capture properties already attempted during open

    # Get frame dimensions
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize configuration
    cfg = CounterConfig(
        top_angle_deg=args.top_angle,
        bottom_angle_deg=args.bottom_angle,
        hold_frames_required=args.hold_frames,
        smoothing_alpha=args.smoothing_alpha,
        min_visibility=args.min_visibility,
        body_alignment_threshold=args.body_alignment_threshold
    )
    
    draw_cfg = DrawConfig(
        show_pose_skeleton=not args.no_skeleton,
        mirror_view=False,  # Disable mirror view to fix left/right detection
        show_debug=args.debug,
    )
    drawer = OverlayDrawer(draw_cfg)

    # Per side state
    states: Dict[str, SideState] = {
        'left': SideState('left'),
        'right': SideState('right'),
    }

    prev_time = time.time()
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, float(args.fps), (frame_w, frame_h))

    window_title = 'flexIntel - Pushup Tracker (Press Q to quit, R to reset)'

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break

            # Unified header
            try:
                from .unified_overlay import draw_header, draw_feedback_panel, draw_fps
            except Exception:
                from unified_overlay import draw_header, draw_feedback_panel, draw_fps
            draw_header(frame, frame_w, "Pushup Tracker")

            if draw_cfg.mirror_view:
                frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                for side in ['left', 'right']:
                    idxs = DICT_FEATURES[side]
                    shldr_lm = landmarks[idxs['shoulder']]
                    elbow_lm = landmarks[idxs['elbow']]
                    wrist_lm = landmarks[idxs['wrist']]
                    hip_lm = landmarks[idxs['hip']]
                    ankle_lm = landmarks[idxs['ankle']]

                    visible_ok = (
                        shldr_lm.visibility >= cfg.min_visibility and
                        elbow_lm.visibility >= cfg.min_visibility and
                        wrist_lm.visibility >= cfg.min_visibility and
                        hip_lm.visibility >= cfg.min_visibility and
                        ankle_lm.visibility >= cfg.min_visibility
                    )

                    if visible_ok:
                        shldr_xy = get_landmark_xy(shldr_lm, frame_w, frame_h)
                        elbow_xy = get_landmark_xy(elbow_lm, frame_w, frame_h)
                        wrist_xy = get_landmark_xy(wrist_lm, frame_w, frame_h)
                        hip_xy = get_landmark_xy(hip_lm, frame_w, frame_h)
                        ankle_xy = get_landmark_xy(ankle_lm, frame_w, frame_h)

                        angle = compute_angle_degrees(shldr_xy, elbow_xy, wrist_xy)
                        states[side].update(angle, shldr_xy, elbow_xy, wrist_xy, hip_xy, ankle_xy, frame_h, cfg)

                        color = states[side].last_color

                        # Draw joints and connections with enhanced styling
                        joint_radius = 8
                        connection_thickness = 3
                        
                        # Draw joints with shadow effect
                        cv2.circle(frame, shldr_xy, joint_radius, (40, 40, 40), -1)
                        cv2.circle(frame, elbow_xy, joint_radius + 1, (40, 40, 40), -1)
                        cv2.circle(frame, wrist_xy, joint_radius, (40, 40, 40), -1)
                        cv2.circle(frame, hip_xy, joint_radius, (40, 40, 40), -1)
                        cv2.circle(frame, ankle_xy, joint_radius, (40, 40, 40), -1)
                        
                        # Draw main joints
                        cv2.circle(frame, shldr_xy, joint_radius, color, -1)
                        cv2.circle(frame, elbow_xy, joint_radius + 1, color, -1)
                        cv2.circle(frame, wrist_xy, joint_radius, color, -1)
                        cv2.circle(frame, hip_xy, joint_radius, color, -1)
                        cv2.circle(frame, ankle_xy, joint_radius, color, -1)
                        
                        # Draw connections with gradient effect
                        cv2.line(frame, shldr_xy, elbow_xy, color, connection_thickness)
                        cv2.line(frame, elbow_xy, wrist_xy, color, connection_thickness)
                        cv2.line(frame, shldr_xy, hip_xy, color, connection_thickness)
                        cv2.line(frame, hip_xy, ankle_xy, color, connection_thickness)

                        # Angle and feedback with professional styling
                        angle_text = f"{angle:.1f}°"
                        cv2.putText(frame, angle_text, (elbow_xy[0] + 15, elbow_xy[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # Draw pose skeleton if enabled
                if draw_cfg.show_pose_skeleton:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=2)
                    )

            # Draw overlays + unified feedback panel
            drawer.draw_rep_counter(frame, states, frame_w, frame_h)
            drawer.draw_form_feedback(frame, states, frame_w, frame_h)
            drawer.draw_progress_bars(frame, states, frame_w, frame_h)

            # Unified feedback: body alignment and depth proxy
            good_alignment = not any(s.last_feedback == "Keep body straight!" for s in states.values())
            good_depth = any(s.last_feedback.startswith("Great!") for s in states.values())
            items = [("Body Aligned", good_alignment), ("Good Depth", good_depth)]
            draw_feedback_panel(frame, items, frame_w, frame_h)

            # FPS
            cur_time = time.time()
            fps = 1.0 / max(cur_time - prev_time, 1e-6)
            prev_time = cur_time
            draw_fps(frame, fps)

            # Debug info with enhanced styling
            if args.debug:
                debug_text = f"Top<{int(cfg.top_angle_deg)}  Bottom>{int(cfg.bottom_angle_deg)}  Hold:{cfg.hold_frames_required}"
                debug_bg_color = (40, 40, 40)
                debug_text_color = (200, 200, 200)
                
                # Draw background rectangle
                debug_text_size = cv2.getTextSize(debug_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(frame, (10, frame_h - 40), (10 + debug_text_size[0] + 20, frame_h - 10), debug_bg_color, -1)
                cv2.rectangle(frame, (10, frame_h - 40), (10 + debug_text_size[0] + 20, frame_h - 10), debug_text_color, 1)
                
                # Draw debug text
                cv2.putText(frame, debug_text, (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1, cv2.LINE_AA)

            if out_writer is not None:
                out_writer.write(frame)

            if not args.no_window:
                # Set window to full screen
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):  # Toggle full screen
                    cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN) != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
                elif key == ord('r'):
                    for state in states.values():
                        state.rep_count = 0
                        state.direction_up = False
                        state.top_hold_frames = 0
                        state.bottom_hold_frames = 0
                        state.ema_angle = None
                        state.body_aligned = False
            else:
                # In headless mode, still allow graceful interruption via frame count or Ctrl+C
                pass

    finally:
        if out_writer is not None:
            out_writer.release()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


def build_arg_parser():
    p = argparse.ArgumentParser(description='Professional Pushup Tracker using MediaPipe Pose')
    p.add_argument('--input', type=str, default='0', help='Video source: camera index (e.g., 0) or path to video file')
    p.add_argument('--width', type=int, default=0, help='Capture width (0 = default)')
    p.add_argument('--height', type=int, default=0, help='Capture height (0 = default)')
    p.add_argument('--fps', type=int, default=30, help='Capture/Output FPS')
    p.add_argument('--output', type=str, default='', help='Optional output path to save annotated video (e.g., output.mp4)')

    p.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2], help='MediaPipe model complexity')
    p.add_argument('--min-detection-confidence', type=float, default=0.5)
    p.add_argument('--min-tracking-confidence', type=float, default=0.5)

    p.add_argument('--top-angle', type=float, default=170.0, help='Top angle threshold (degrees)')
    p.add_argument('--bottom-angle', type=float, default=90.0, help='Bottom angle threshold (degrees)')
    p.add_argument('--hold-frames', type=int, default=3, help='Frames to hold thresholds to register state')
    p.add_argument('--smoothing-alpha', type=float, default=0.2, help='EMA smoothing factor for angle')
    p.add_argument('--min-visibility', type=float, default=0.6, help='Minimum landmark visibility to use joints')
    p.add_argument('--body-alignment-threshold', type=float, default=0.1, help='Body alignment threshold')

    p.add_argument('--no-skeleton', action='store_true', help='Disable drawing of full pose skeleton')
    p.add_argument('--no-mirror', action='store_true', help='Disable mirror view (selfie)')
    p.add_argument('--debug', action='store_true', help='Show debug info overlay')
    p.add_argument('--no-window', action='store_true', help='Run headless (no GUI window)')
    return p


def parse_input_source(input_arg: str):
    # Try to parse as int for camera index, otherwise use as path
    try:
        return int(input_arg)
    except ValueError:
        return input_arg


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    args.input = parse_input_source(args.input)
    run_app(args)
