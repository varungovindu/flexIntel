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
    top_angle_deg: float = 85.0            # Shoulder abduction angle considered top of raise
    bottom_angle_deg: float = 25.0         # Shoulder abduction angle considered bottom (arms by side)
    hold_frames_required: int = 3          # Frames to hold top/bottom to register state
    smoothing_alpha: float = 0.2           # EMA smoothing factor for shoulder angle
    min_visibility: float = 0.6            # Min landmark visibility to be considered valid
    elbow_lead_check: bool = True          # Warn if wrist gets higher than elbow at top
    shrug_warn_ratio: float = 0.04         # Warn if shoulder elevates upward > ratio of frame height


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

# Landmark indices for arms and torso
DICT_FEATURES = {
    'left': {
        'hip': mp_pose.PoseLandmark.LEFT_HIP.value,
        'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        'elbow': mp_pose.PoseLandmark.LEFT_ELBOW.value,
        'wrist': mp_pose.PoseLandmark.LEFT_WRIST.value,
    },
    'right': {
        'hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
        'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        'wrist': mp_pose.PoseLandmark.RIGHT_WRIST.value,
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
    last_feedback: str = ""
    last_color: Tuple[int, int, int] = (255, 255, 255)
    progress_01: float = 0.0
    initial_shoulder_y: Optional[int] = None
    shrug_warn: bool = False
    elbow_lead_warn: bool = False

    def update(self,
               shoulder_angle_raw: Optional[float],
               shoulder_xy: Tuple[int, int],
               elbow_xy: Tuple[int, int],
               wrist_xy: Tuple[int, int],
               frame_h: int,
               cfg: CounterConfig) -> None:
        if shoulder_angle_raw is None:
            self.last_feedback = "Landmarks unstable"
            self.last_color = (0, 165, 255)
            return

        # EMA smoothing
        if self.ema_angle is None:
            self.ema_angle = shoulder_angle_raw
        else:
            self.ema_angle = cfg.smoothing_alpha * shoulder_angle_raw + (1.0 - cfg.smoothing_alpha) * self.ema_angle

        angle = float(self.ema_angle)

        # Initialize shoulder baseline
        if self.initial_shoulder_y is None:
            self.initial_shoulder_y = shoulder_xy[1]

        # Progress [0..1]: 0 at bottom, 1 at top
        angle_clamped = np.clip(angle, cfg.bottom_angle_deg, cfg.top_angle_deg)
        denom = max(cfg.top_angle_deg - cfg.bottom_angle_deg, 1e-3)
        self.progress_01 = float((angle_clamped - cfg.bottom_angle_deg) / denom)

        # Hysteresis
        if angle >= cfg.top_angle_deg:
            self.top_hold_frames += 1
            self.bottom_hold_frames = 0
        elif angle <= cfg.bottom_angle_deg:
            self.bottom_hold_frames += 1
            self.top_hold_frames = 0
        else:
            self.top_hold_frames = 0
            self.bottom_hold_frames = 0

        if self.top_hold_frames >= cfg.hold_frames_required:
            self.direction_up = True

        if self.bottom_hold_frames >= cfg.hold_frames_required and self.direction_up:
            self.rep_count += 1
            self.direction_up = False

        # Feedback by angle
        if angle >= cfg.top_angle_deg:
            self.last_feedback = "Good height"
            self.last_color = (0, 200, 0)
        elif angle <= cfg.bottom_angle_deg:
            self.last_feedback = "Lower fully"
            self.last_color = (0, 255, 255)
        else:
            self.last_feedback = "Raise higher"
            self.last_color = (0, 255, 255)

        # Elbow lead check near top
        self.elbow_lead_warn = False
        if cfg.elbow_lead_check and self.progress_01 > 0.75:
            # y increases downward; elbow should be slightly higher (smaller y) than wrist
            if wrist_xy[1] < elbow_xy[1]:
                self.elbow_lead_warn = True

        # Shoulder shrug warning: if shoulder moved up more than threshold
        self.shrug_warn = False
        if self.initial_shoulder_y is not None:
            upward_px = max(0, self.initial_shoulder_y - shoulder_xy[1])  # positive if moved up
            if upward_px / float(max(frame_h, 1)) > cfg.shrug_warn_ratio:
                self.shrug_warn = True


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
        elbow_lead_check=not args.no_elbow_lead_check,
        shrug_warn_ratio=args.shrug_warn_ratio,
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

    window_title = 'Lateral Raise Tracker (press q to quit, r to reset)'

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

                for side in ['left', 'right']:
                    idxs = DICT_FEATURES[side]
                    hip_lm = lms[idxs['hip']]
                    shldr_lm = lms[idxs['shoulder']]
                    elbow_lm = lms[idxs['elbow']]
                    wrist_lm = lms[idxs['wrist']]

                    visible_ok = (
                        hip_lm.visibility >= cfg.min_visibility and
                        shldr_lm.visibility >= cfg.min_visibility and
                        elbow_lm.visibility >= cfg.min_visibility and
                        wrist_lm.visibility >= cfg.min_visibility
                    )

                    label_y = 40 if side == 'left' else 80
                    if not visible_ok:
                        drawer.put_text(frame, f"{side.capitalize()} arm not visible", (10, label_y), (0, 165, 255))
                        continue

                    hip_xy = get_landmark_xy(hip_lm, frame_w, frame_h)
                    shldr_xy = get_landmark_xy(shldr_lm, frame_w, frame_h)
                    elbow_xy = get_landmark_xy(elbow_lm, frame_w, frame_h)
                    wrist_xy = get_landmark_xy(wrist_lm, frame_w, frame_h)

                    # Shoulder abduction angle: angle at shoulder between hip and elbow
                    shoulder_angle = compute_angle_degrees(hip_xy, shldr_xy, elbow_xy)

                    states[side].update(shoulder_angle, shldr_xy, elbow_xy, wrist_xy, frame_h, cfg)

                    color = states[side].last_color

                    # Draw joints and connections
                    cv2.circle(frame, hip_xy, 7, color, -1)
                    cv2.circle(frame, shldr_xy, 9, color, -1)
                    cv2.circle(frame, elbow_xy, 7, color, -1)
                    cv2.circle(frame, wrist_xy, 7, color, -1)
                    cv2.line(frame, hip_xy, shldr_xy, color, 3)
                    cv2.line(frame, shldr_xy, elbow_xy, color, 3)
                    cv2.line(frame, elbow_xy, wrist_xy, color, 3)

                    # Angle text and feedback
                    if states[side].ema_angle is not None:
                        drawer.put_text(frame, f"{int(states[side].ema_angle)}°", (shldr_xy[0] + 10, shldr_xy[1] - 12), color)
                    drawer.put_text(frame, states[side].last_feedback, (shldr_xy[0] + 10, shldr_xy[1] + 20), color)

                    if states[side].elbow_lead_warn:
                        drawer.put_text(frame, "Lead with elbows", (shldr_xy[0] + 10, shldr_xy[1] + 44), (0, 140, 255))
                    if states[side].shrug_warn:
                        drawer.put_text(frame, "Don't shrug", (shldr_xy[0] + 10, shldr_xy[1] + 68), (0, 165, 255))

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
                                f"Top>{int(cfg.top_angle_deg)}  Bottom<{int(cfg.bottom_angle_deg)}  Hold:{cfg.hold_frames_required}",
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
                        state.ema_angle = None
                        state.initial_shoulder_y = None
                        state.shrug_warn = False
                        state.elbow_lead_warn = False
            else:
                pass

    finally:
        if out_writer is not None:
            out_writer.release()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


def build_arg_parser():
    p = argparse.ArgumentParser(description='Lateral Raise Tracker using MediaPipe Pose')
    p.add_argument('--input', type=str, default='0', help='Video source: camera index (e.g., 0) or path to video file')
    p.add_argument('--width', type=int, default=0, help='Capture width (0 = default)')
    p.add_argument('--height', type=int, default=0, help='Capture height (0 = default)')
    p.add_argument('--fps', type=int, default=30, help='Capture/Output FPS')
    p.add_argument('--output', type=str, default='', help='Optional output path to save annotated video (e.g., output.mp4)')

    p.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2], help='MediaPipe model complexity')
    p.add_argument('--min-detection-confidence', type=float, default=0.5)
    p.add_argument('--min-tracking-confidence', type=float, default=0.5)

    p.add_argument('--top-angle', type=float, default=85.0, help='Top angle threshold (degrees)')
    p.add_argument('--bottom-angle', type=float, default=25.0, help='Bottom angle threshold (degrees)')
    p.add_argument('--hold-frames', type=int, default=3, help='Frames to hold thresholds to register state')
    p.add_argument('--smoothing-alpha', type=float, default=0.2, help='EMA smoothing factor for angle')
    p.add_argument('--min-visibility', type=float, default=0.6, help='Minimum landmark visibility to use joints')

    p.add_argument('--no-skeleton', action='store_true', help='Disable drawing of full pose skeleton')
    p.add_argument('--no-mirror', action='store_true', help='Disable mirror view (selfie)')
    p.add_argument('--debug', action='store_true', help='Show debug info overlay')
    p.add_argument('--no-window', action='store_true', help='Run headless (no GUI window)')

    p.add_argument('--no-elbow-lead-check', action='store_true', help='Disable elbow lead form check near top')
    p.add_argument('--shrug-warn-ratio', type=float, default=0.04, help='Warn if shoulder elevates upward > ratio*frame_h')
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