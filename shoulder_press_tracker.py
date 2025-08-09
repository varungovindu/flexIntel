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
    top_offset_ratio: float = 0.07        # Portion of frame height above head (nose) that defines top threshold
    bottom_offset_ratio: float = 0.08     # Portion of frame height below shoulder that defines bottom threshold
    hold_frames_required: int = 3         # Frames to hold top/bottom to register state
    smoothing_alpha: float = 0.2          # EMA smoothing factor for wrist y
    min_visibility: float = 0.6           # Min landmark visibility to be considered valid
    forearm_vertical_tolerance_deg: float = 25.0  # Max deviation from vertical for forearm


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

# Landmark indices for arms and head
DICT_FEATURES = {
    'left': {
        'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        'elbow': mp_pose.PoseLandmark.LEFT_ELBOW.value,
        'wrist': mp_pose.PoseLandmark.LEFT_WRIST.value,
    },
    'right': {
        'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        'wrist': mp_pose.PoseLandmark.RIGHT_WRIST.value,
    },
}
HEAD_LANDMARK_INDEX = mp_pose.PoseLandmark.NOSE.value


def get_landmark_xy(landmark, frame_w: int, frame_h: int) -> Tuple[int, int]:
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def compute_angle_with_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> Optional[float]:
    # Angle between vector p2->p1 and vertical up vector (0,-1)
    vx, vy = (p1[0] - p2[0]), (p1[1] - p2[1])
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
    ema_wrist_y: Optional[float] = None
    top_hold_frames: int = 0
    bottom_hold_frames: int = 0
    last_feedback: str = ""
    last_color: Tuple[int, int, int] = (255, 255, 255)
    progress_01: float = 0.0

    def update(self,
               wrist_y_raw: Optional[float],
               top_threshold_y: float,
               bottom_threshold_y: float,
               forearm_angle_deg: Optional[float],
               cfg: CounterConfig) -> None:
        # Validate wrist position
        if wrist_y_raw is None:
            self.last_feedback = "Landmarks unstable"
            self.last_color = (0, 165, 255)
            return

        # EMA smoothing on wrist y (remember: y increases downwards)
        if self.ema_wrist_y is None:
            self.ema_wrist_y = wrist_y_raw
        else:
            self.ema_wrist_y = cfg.smoothing_alpha * wrist_y_raw + (1.0 - cfg.smoothing_alpha) * self.ema_wrist_y

        wy = float(self.ema_wrist_y)

        # Hysteresis thresholds: top is smaller y (up), bottom is larger y (down)
        if wy <= top_threshold_y:
            self.top_hold_frames += 1
            self.bottom_hold_frames = 0
        elif wy >= bottom_threshold_y:
            self.bottom_hold_frames += 1
            self.top_hold_frames = 0
        else:
            self.top_hold_frames = 0
            self.bottom_hold_frames = 0

        # Enter up direction when we have held the top
        if self.top_hold_frames >= cfg.hold_frames_required:
            self.direction_up = True

        # Count a rep when we have been at bottom after being up
        if self.bottom_hold_frames >= cfg.hold_frames_required and self.direction_up:
            self.rep_count += 1
            self.direction_up = False

        # Progress [0..1] from bottom to top. Clamp to avoid div by zero
        denom = max(bottom_threshold_y - top_threshold_y, 1.0)
        self.progress_01 = float(np.clip((bottom_threshold_y - wy) / denom, 0.0, 1.0))

        # Feedback
        if wy > bottom_threshold_y + 2:  # not reaching bottom
            self.last_feedback = "Lower to shoulder"
            self.last_color = (0, 255, 255)
        elif wy < top_threshold_y - 2:  # beyond top
            self.last_feedback = "Good lockout"
            self.last_color = (0, 200, 0)
        else:
            self.last_feedback = "Press higher"
            self.last_color = (0, 255, 255)

        if forearm_angle_deg is not None and forearm_angle_deg > cfg.forearm_vertical_tolerance_deg:
            self.last_feedback = "Keep forearm vertical"
            self.last_color = (0, 165, 255)


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
        top_offset_ratio=args.top_offset_ratio,
        bottom_offset_ratio=args.bottom_offset_ratio,
        hold_frames_required=args.hold_frames,
        smoothing_alpha=args.smoothing_alpha,
        min_visibility=args.min_visibility,
        forearm_vertical_tolerance_deg=args.vertical_forearm_tolerance,
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

    window_title = 'Shoulder Press Tracker (press q to quit, r to reset)'

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

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                # Head height (nose)
                nose_lm = lms[HEAD_LANDMARK_INDEX]
                head_visible = nose_lm.visibility >= cfg.min_visibility
                nose_xy = get_landmark_xy(nose_lm, frame_w, frame_h) if head_visible else (frame_w // 2, int(frame_h * 0.2))

                for side in ['left', 'right']:
                    idxs = DICT_FEATURES[side]
                    shldr_lm = lms[idxs['shoulder']]
                    elbow_lm = lms[idxs['elbow']]
                    wrist_lm = lms[idxs['wrist']]

                    visible_ok = (
                        shldr_lm.visibility >= cfg.min_visibility and
                        elbow_lm.visibility >= cfg.min_visibility and
                        wrist_lm.visibility >= cfg.min_visibility
                    )

                    label_y = 40 if side == 'left' else 80
                    if not visible_ok:
                        drawer.put_text(frame, f"{side.capitalize()} arm not visible", (10, label_y), (0, 165, 255))
                        continue

                    shldr_xy = get_landmark_xy(shldr_lm, frame_w, frame_h)
                    elbow_xy = get_landmark_xy(elbow_lm, frame_w, frame_h)
                    wrist_xy = get_landmark_xy(wrist_lm, frame_w, frame_h)

                    # Thresholds
                    top_thresh_y = max(nose_xy[1] - int(cfg.top_offset_ratio * frame_h), 0)
                    bottom_thresh_y = min(shldr_xy[1] + int(cfg.bottom_offset_ratio * frame_h), frame_h - 1)

                    # Forearm verticality
                    forearm_angle_deg = compute_angle_with_vertical(wrist_xy, elbow_xy)

                    states[side].update(wrist_y_raw=float(wrist_xy[1]),
                                        top_threshold_y=float(top_thresh_y),
                                        bottom_threshold_y=float(bottom_thresh_y),
                                        forearm_angle_deg=forearm_angle_deg,
                                        cfg=cfg)

                    color = states[side].last_color

                    # Draw joints and connections
                    cv2.circle(frame, shldr_xy, 7, color, -1)
                    cv2.circle(frame, elbow_xy, 9, color, -1)
                    cv2.circle(frame, wrist_xy, 7, color, -1)
                    cv2.line(frame, shldr_xy, elbow_xy, color, 3)
                    cv2.line(frame, elbow_xy, wrist_xy, color, 3)

                    # Threshold guides
                    if args.guides:
                        cv2.line(frame, (0, top_thresh_y), (frame_w, top_thresh_y), (80, 200, 80), 1)
                        cv2.line(frame, (0, bottom_thresh_y), (frame_w, bottom_thresh_y), (80, 80, 200), 1)

                    # Feedback
                    drawer.put_text(frame, states[side].last_feedback, (elbow_xy[0] + 10, elbow_xy[1] + 20), color)
                    if forearm_angle_deg is not None and args.debug:
                        drawer.put_text(frame, f"Forearm Δ: {int(forearm_angle_deg)}°", (elbow_xy[0] + 10, elbow_xy[1] + 44), (200, 200, 200), 0.6, 1)

                    # Progress bar
                    bar_w, bar_h = 18, int(frame_h * 0.6)
                    margin = 16
                    x = margin if side == 'left' else (frame_w - margin - bar_w)
                    y = int((frame_h - bar_h) / 2)
                    OverlayDrawer.draw_progress_bar(frame, x, y, bar_w, bar_h, states[side].progress_01, color)

                    # Rep count
                    drawer.put_text(frame, f"{side.capitalize()} Reps: {states[side].rep_count}", (10, label_y))

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
                                f"TopOff:{cfg.top_offset_ratio:.2f}  BotOff:{cfg.bottom_offset_ratio:.2f}  Hold:{cfg.hold_frames_required}",
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
                        state.ema_wrist_y = None
            else:
                pass

    finally:
        if out_writer is not None:
            out_writer.release()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


def build_arg_parser():
    p = argparse.ArgumentParser(description='Shoulder Press Tracker using MediaPipe Pose')
    p.add_argument('--input', type=str, default='0', help='Video source: camera index (e.g., 0) or path to video file')
    p.add_argument('--width', type=int, default=0, help='Capture width (0 = default)')
    p.add_argument('--height', type=int, default=0, help='Capture height (0 = default)')
    p.add_argument('--fps', type=int, default=30, help='Capture/Output FPS')
    p.add_argument('--output', type=str, default='', help='Optional output path to save annotated video (e.g., output.mp4)')

    p.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2], help='MediaPipe model complexity')
    p.add_argument('--min-detection-confidence', type=float, default=0.5)
    p.add_argument('--min-tracking-confidence', type=float, default=0.5)

    p.add_argument('--top-offset-ratio', type=float, default=0.07, help='Top threshold = nose_y - ratio*frame_h')
    p.add_argument('--bottom-offset-ratio', type=float, default=0.08, help='Bottom threshold = shoulder_y + ratio*frame_h')
    p.add_argument('--hold-frames', type=int, default=3, help='Frames to hold thresholds to register state')
    p.add_argument('--smoothing-alpha', type=float, default=0.2, help='EMA smoothing factor for wrist y')
    p.add_argument('--min-visibility', type=float, default=0.6, help='Minimum landmark visibility to use joints')
    p.add_argument('--vertical-forearm-tolerance', type=float, default=25.0, help='Max deviation from vertical (deg)')

    p.add_argument('--no-skeleton', action='store_true', help='Disable drawing of full pose skeleton')
    p.add_argument('--no-mirror', action='store_true', help='Disable mirror view (selfie)')
    p.add_argument('--debug', action='store_true', help='Show debug info overlay')
    p.add_argument('--no-window', action='store_true', help='Run headless (no GUI window)')
    p.add_argument('--guides', action='store_true', help='Draw top/bottom threshold guide lines')
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