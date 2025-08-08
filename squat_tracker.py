import argparse
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# -------------------------------
# Configuration dataclasses
# -------------------------------
@dataclass
class CounterConfig:
    top_angle_deg: float = 85.0            # Knee angle at bottom of squat (smaller angle)
    bottom_angle_deg: float = 170.0        # Knee angle at top/standing (larger angle)
    hold_frames_required: int = 3          # Frames to hold top/bottom to register state
    smoothing_alpha: float = 0.25          # EMA smoothing factor for knee angle
    min_visibility: float = 0.6            # Min landmark visibility to be considered valid
    torso_forward_tolerance_deg: float = 35.0  # Max torso lean angle relative to vertical
    knee_cave_ratio_warn: float = 0.8      # Warn if knee distance < ratio * ankle distance


@dataclass
class DrawConfig:
    show_pose_skeleton: bool = True
    mirror_view: bool = True
    show_debug: bool = False


# -------------------------------
# Pose + utility helpers
# -------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Landmark indices
DICT_FEATURES = {
    'left': {
        'hip': mp_pose.PoseLandmark.LEFT_HIP.value,
        'knee': mp_pose.PoseLandmark.LEFT_KNEE.value,
        'ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value,
        'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    },
    'right': {
        'hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
        'knee': mp_pose.PoseLandmark.RIGHT_KNEE.value,
        'ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    },
}


def get_landmark_xy(landmark, frame_w: int, frame_h: int) -> Tuple[int, int]:
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def compute_angle_degrees(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> Optional[float]:
    # angle at b from vectors ba and bc
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


def compute_angle_with_vertical(p_top: Tuple[int, int], p_bottom: Tuple[int, int]) -> Optional[float]:
    # Angle between vector bottom->top and vertical up vector (0,-1)
    vx, vy = (p_top[0] - p_bottom[0]), (p_top[1] - p_bottom[1])
    v = np.array([vx, vy], dtype=np.float32)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-6:
        return None
    v_unit = v / norm_v
    vertical_up = np.array([0.0, -1.0], dtype=np.float32)
    cos_angle = float(np.dot(v_unit, vertical_up))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    ang_rad = np.arccos(cos_angle)
    return float(np.degrees(ang_rad))


# -------------------------------
# Rep counting state machine
# -------------------------------
@dataclass
class SideState:
    side_name: str
    rep_count: int = 0
    direction_up: bool = False  # False = going down, True = going up
    ema_knee_angle: Optional[float] = None
    top_hold_frames: int = 0
    bottom_hold_frames: int = 0
    last_feedback: str = ""
    last_color: Tuple[int, int, int] = (255, 255, 255)
    progress_01: float = 0.0

    def update(self,
               knee_angle_raw: Optional[float],
               cfg: CounterConfig) -> None:
        if knee_angle_raw is None:
            self.last_feedback = "Landmarks unstable"
            self.last_color = (0, 165, 255)
            return

        # EMA smoothing
        if self.ema_knee_angle is None:
            self.ema_knee_angle = knee_angle_raw
        else:
            self.ema_knee_angle = cfg.smoothing_alpha * knee_angle_raw + (1.0 - cfg.smoothing_alpha) * self.ema_knee_angle

        angle = float(self.ema_knee_angle)

        # Progress [0..1]: 0 at top (standing), 1 at bottom (deep)
        angle_clamped = np.clip(angle, cfg.top_angle_deg, cfg.bottom_angle_deg)
        denom = max(cfg.bottom_angle_deg - cfg.top_angle_deg, 1e-3)
        self.progress_01 = float((cfg.bottom_angle_deg - angle_clamped) / denom)

        # Hysteresis
        if angle <= cfg.top_angle_deg:
            self.top_hold_frames += 1
            self.bottom_hold_frames = 0
        elif angle >= cfg.bottom_angle_deg:
            self.bottom_hold_frames += 1
            self.top_hold_frames = 0
        else:
            self.top_hold_frames = 0
            self.bottom_hold_frames = 0

        # Set direction at bottom
        if self.top_hold_frames >= cfg.hold_frames_required:
            self.direction_up = True

        # Count when we reach top after being at bottom
        if self.bottom_hold_frames >= cfg.hold_frames_required and self.direction_up:
            self.rep_count += 1
            self.direction_up = False

        # Feedback by angle
        if angle <= cfg.top_angle_deg:
            self.last_feedback = "Good depth"
            self.last_color = (0, 200, 0)
        elif angle >= cfg.bottom_angle_deg:
            self.last_feedback = "Stand tall"
            self.last_color = (0, 255, 255)
        else:
            self.last_feedback = "Go deeper"
            self.last_color = (0, 255, 255)


# -------------------------------
# Drawing helpers
# -------------------------------
class OverlayDrawer:
    def __init__(self, draw_cfg: DrawConfig):
        self.draw_cfg = draw_cfg

    @staticmethod
    def draw_progress_bar(image: np.ndarray, x: int, y: int, width: int, height: int, progress_01: float,
                          color: Tuple[int, int, int]) -> None:
        progress_01 = float(np.clip(progress_01, 0.0, 1.0))
        cv2.rectangle(image, (x, y), (x + width, y + height), (60, 60, 60), 2)
        inner_h = int(height * progress_01)
        cv2.rectangle(image, (x + 2, y + height - inner_h), (x + width - 2, y + height - 2), color, -1)

    @staticmethod
    def put_text(image: np.ndarray, text: str, org: Tuple[int, int], color=(255, 255, 255), scale=0.7, thickness=2):
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# -------------------------------
# Main application
# -------------------------------

def run_app(args):
    cap = cv2.VideoCapture(args.input)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("Failed to access video source.")
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        print("Failed to read first frame.")
        cap.release()
        return

    frame_h, frame_w = frame.shape[:2]

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    cfg = CounterConfig(
        top_angle_deg=args.top_angle,
        bottom_angle_deg=args.bottom_angle,
        hold_frames_required=args.hold_frames,
        smoothing_alpha=args.smoothing_alpha,
        min_visibility=args.min_visibility,
        torso_forward_tolerance_deg=args.torso_forward_tolerance,
        knee_cave_ratio_warn=args.knee_cave_ratio,
    )
    draw_cfg = DrawConfig(
        show_pose_skeleton=not args.no_skeleton,
        mirror_view=not args.no_mirror,
        show_debug=args.debug,
    )
    drawer = OverlayDrawer(draw_cfg)

    states: Dict[str, SideState] = {
        'left': SideState('left'),
        'right': SideState('right'),
    }

    prev_time = time.time()
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, float(args.fps), (frame_w, frame_h))

    window_title = 'Squat Tracker (press q to quit, r to reset)'

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if draw_cfg.mirror_view:
                frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            knees_visible = False
            ankles_visible = False
            knees_x = []
            ankles_x = []

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                for side in ['left', 'right']:
                    idxs = DICT_FEATURES[side]
                    hip_lm = lms[idxs['hip']]
                    knee_lm = lms[idxs['knee']]
                    ankle_lm = lms[idxs['ankle']]
                    shoulder_lm = lms[idxs['shoulder']]

                    visible_ok = (
                        hip_lm.visibility >= cfg.min_visibility and
                        knee_lm.visibility >= cfg.min_visibility and
                        ankle_lm.visibility >= cfg.min_visibility and
                        shoulder_lm.visibility >= cfg.min_visibility
                    )

                    label_y = 40 if side == 'left' else 80
                    if not visible_ok:
                        drawer.put_text(frame, f"{side.capitalize()} leg not visible", (10, label_y), (0, 165, 255))
                        continue

                    hip_xy = get_landmark_xy(hip_lm, frame_w, frame_h)
                    knee_xy = get_landmark_xy(knee_lm, frame_w, frame_h)
                    ankle_xy = get_landmark_xy(ankle_lm, frame_w, frame_h)
                    shoulder_xy = get_landmark_xy(shoulder_lm, frame_w, frame_h)

                    # Record for global valgus check
                    knees_visible = True
                    ankles_visible = True
                    knees_x.append(knee_xy[0])
                    ankles_x.append(ankle_xy[0])

                    # Knee angle
                    knee_angle = compute_angle_degrees(hip_xy, knee_xy, ankle_xy)
                    states[side].update(knee_angle, cfg)

                    color = states[side].last_color

                    # Draw joints and connections
                    cv2.circle(frame, hip_xy, 7, color, -1)
                    cv2.circle(frame, knee_xy, 9, color, -1)
                    cv2.circle(frame, ankle_xy, 7, color, -1)
                    cv2.line(frame, hip_xy, knee_xy, color, 3)
                    cv2.line(frame, knee_xy, ankle_xy, color, 3)

                    # Angle text and feedback
                    if states[side].ema_knee_angle is not None:
                        drawer.put_text(frame, f"{int(states[side].ema_knee_angle)}°", (knee_xy[0] + 10, knee_xy[1] - 12), color)
                    drawer.put_text(frame, states[side].last_feedback, (knee_xy[0] + 10, knee_xy[1] + 20), color)

                    # Torso angle warning (shoulder->hip against vertical)
                    torso_angle = compute_angle_with_vertical(shoulder_xy, hip_xy)
                    if torso_angle is not None and torso_angle > cfg.torso_forward_tolerance_deg:
                        drawer.put_text(frame, "Chest up", (hip_xy[0] + 10, hip_xy[1] + 44), (0, 140, 255))

                    # Progress bar
                    bar_w, bar_h = 18, int(frame_h * 0.6)
                    margin = 16
                    x = margin if side == 'left' else (frame_w - margin - bar_w)
                    y = int((frame_h - bar_h) / 2)
                    OverlayDrawer.draw_progress_bar(frame, x, y, bar_w, bar_h, states[side].progress_01, color)

                    # Rep count
                    drawer.put_text(frame, f"{side.capitalize()} Reps: {states[side].rep_count}", (10, label_y))

                # Global knee valgus check: compare knee width vs ankle width
                if args.valgus_warning and knees_visible and ankles_visible and len(knees_x) == 2 and len(ankles_x) == 2:
                    knee_width = abs(knees_x[1] - knees_x[0])
                    ankle_width = abs(ankles_x[1] - ankles_x[0])
                    if ankle_width > 1 and knee_width < cfg.knee_cave_ratio_warn * ankle_width:
                        drawer.put_text(frame, "Knees caving in", (10, 120), (0, 165, 255))

                if draw_cfg.show_pose_skeleton:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                drawer.put_text(frame, "No person detected", (10, 40), (0, 165, 255))

            # FPS
            cur_time = time.time()
            fps = 1.0 / max(cur_time - prev_time, 1e-6)
            prev_time = cur_time
            drawer.put_text(frame, f'FPS: {int(fps)}', (10, 20), (100, 255, 0))

            if args.debug:
                drawer.put_text(frame,
                                f"Top<{int(cfg.top_angle_deg)}  Bottom>{int(cfg.bottom_angle_deg)}  Hold:{cfg.hold_frames_required}",
                                (10, frame_h - 20), (200, 200, 200), 0.6, 1)

            if out_writer is not None:
                out_writer.write(frame)

            if not args.no_window:
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    for state in states.values():
                        state.rep_count = 0
                        state.direction_up = False
                        state.top_hold_frames = 0
                        state.bottom_hold_frames = 0
                        state.ema_knee_angle = None
            else:
                pass

    finally:
        if out_writer is not None:
            out_writer.release()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


def build_arg_parser():
    p = argparse.ArgumentParser(description='Squat Tracker using MediaPipe Pose')
    p.add_argument('--input', type=str, default='0', help='Video source: camera index (e.g., 0) or path to video file')
    p.add_argument('--width', type=int, default=0, help='Capture width (0 = default)')
    p.add_argument('--height', type=int, default=0, help='Capture height (0 = default)')
    p.add_argument('--fps', type=int, default=30, help='Capture/Output FPS')
    p.add_argument('--output', type=str, default='', help='Optional output path to save annotated video (e.g., output.mp4)')

    p.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2], help='MediaPipe model complexity')
    p.add_argument('--min-detection-confidence', type=float, default=0.5)
    p.add_argument('--min-tracking-confidence', type=float, default=0.5)

    p.add_argument('--top-angle', type=float, default=85.0, help='Bottom-of-squat knee angle threshold (degrees)')
    p.add_argument('--bottom-angle', type=float, default=170.0, help='Top/standing knee angle threshold (degrees)')
    p.add_argument('--hold-frames', type=int, default=3, help='Frames to hold thresholds to register state')
    p.add_argument('--smoothing-alpha', type=float, default=0.25, help='EMA smoothing factor for knee angle')
    p.add_argument('--min-visibility', type=float, default=0.6, help='Minimum landmark visibility to use joints')

    p.add_argument('--torso-forward-tolerance', type=float, default=35.0, help='Max torso lean from vertical (deg)')
    p.add_argument('--knee-cave-ratio', type=float, default=0.8, help='Warn if knee width < ratio * ankle width')

    p.add_argument('--no-skeleton', action='store_true', help='Disable drawing of full pose skeleton')
    p.add_argument('--no-mirror', action='store_true', help='Disable mirror view (selfie)')
    p.add_argument('--debug', action='store_true', help='Show debug info overlay')
    p.add_argument('--no-window', action='store_true', help='Run headless (no GUI window)')
    p.add_argument('--valgus-warning', action='store_true', help='Enable global knees-in warning')
    return p


def parse_input_source(input_arg: str):
    try:
        return int(input_arg)
    except ValueError:
        return input_arg


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    args.input = parse_input_source(args.input)
    run_app(args)