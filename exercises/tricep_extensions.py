import argparse
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, cast

import cv2
import mediapipe as mp
import numpy as np

solutions: Any = mp.solutions
mp_pose: Any = solutions.pose
mp_drawing: Any = solutions.drawing_utils
cv2 = cast(Any, cv2)

@dataclass
class CounterConfig:
    top_angle_deg: float = 30.0     # Elbow angle at full extension (smaller angle when overhead)
    bottom_angle_deg: float = 110.0 # Elbow angle at bottom (forearm down behind head)
    hold_frames_required: int = 3
    smoothing_alpha: float = 0.2
    min_visibility: float = 0.6

@dataclass
class DrawConfig:
    show_pose_skeleton: bool = True
    mirror_view: bool = False
    show_debug: bool = False

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

def open_video_capture(input_source, width: int, height: int, fps: int) -> Any:
    backends = [getattr(cv2, 'CAP_DSHOW', 700), getattr(cv2, 'CAP_MSMF', 1400), getattr(cv2, 'CAP_ANY', 0)] if isinstance(input_source, int) else [getattr(cv2, 'CAP_ANY', 0)]
    for backend in backends:
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
        try:
            cap.release()
        except Exception:
            pass
    cap = cv2.VideoCapture(input_source)
    if width and width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


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

@dataclass
class SideState:
    side_name: str
    rep_count: int = 0
    direction_up: bool = False
    ema_angle: Optional[float] = None
    top_hold_frames: int = 0
    bottom_hold_frames: int = 0
    last_feedback: str = ''
    last_color: Tuple[int, int, int] = (255, 255, 255)
    progress_01: float = 0.0

    def update(self, elbow_angle_raw: Optional[float], cfg: CounterConfig) -> None:
        if elbow_angle_raw is None:
            self.last_feedback = 'Landmarks unstable'
            self.last_color = (0, 165, 255)
            return
        if self.ema_angle is None:
            self.ema_angle = elbow_angle_raw
        else:
            self.ema_angle = cfg.smoothing_alpha * elbow_angle_raw + (1.0 - cfg.smoothing_alpha) * self.ema_angle
        angle = float(self.ema_angle)
        # Progress: 0 at top (full extension), 1 at bottom
        angle_clamped = np.clip(angle, cfg.top_angle_deg, cfg.bottom_angle_deg)
        denom = max(cfg.bottom_angle_deg - cfg.top_angle_deg, 1e-3)
        self.progress_01 = float((angle_clamped - cfg.top_angle_deg) / denom)
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
        # Count rep when reaching top after being at bottom
        if self.bottom_hold_frames >= cfg.hold_frames_required:
            self.direction_up = True
        if self.top_hold_frames >= cfg.hold_frames_required and self.direction_up:
            self.rep_count += 1
            self.direction_up = False
        # Feedback
        if angle <= cfg.top_angle_deg:
            self.last_feedback = 'Full extension'
            self.last_color = (0, 200, 0)
        elif angle >= cfg.bottom_angle_deg:
            self.last_feedback = 'Lower fully'
            self.last_color = (0, 255, 255)
        else:
            self.last_feedback = 'Extend higher'
            self.last_color = (0, 255, 255)

class OverlayDrawer:
    def __init__(self, cfg: DrawConfig):
        self.cfg = cfg

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

    @staticmethod
    def draw_rep_counter(image: np.ndarray, x: int, y: int, rep_count: int, side: str, color: Tuple[int, int, int]):
        bg_width, bg_height = 120, 40
        cv2.rectangle(image, (x, y), (x + bg_width, y + bg_height), (40, 40, 40), -1)
        cv2.rectangle(image, (x, y), (x + bg_width, y + bg_height), color, 2)
        cv2.putText(image, f"{side.upper()}", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, f"{rep_count}", (x + 5, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def run_app(args):
    cap = open_video_capture(args.input, args.width, args.height, args.fps)
    if not cap.isOpened():
        print('Failed to access video source.')
        return
    ok, frame = cap.read()
    if not ok or frame is None:
        print('Failed to read first frame.')
        cap.release()
        return

    frame_h, frame_w = frame.shape[:2]
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=args.model_complexity,
                        smooth_landmarks=True,
                        min_detection_confidence=args.min_detection_confidence,
                        min_tracking_confidence=args.min_tracking_confidence)

    cfg = CounterConfig(top_angle_deg=args.top_angle,
                        bottom_angle_deg=args.bottom_angle,
                        hold_frames_required=args.hold_frames,
                        smoothing_alpha=args.smoothing_alpha,
                        min_visibility=args.min_visibility)
    draw_cfg = DrawConfig(show_pose_skeleton=not args.no_skeleton,
                          mirror_view=False,
                          show_debug=args.debug)

    states: Dict[str, SideState] = {'left': SideState('left'), 'right': SideState('right')}
    drawer = OverlayDrawer(draw_cfg)

    prev_time = time.time()
    window_title = 'flexIntel - Tricep Extensions (press q to quit)'

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
                    elbow_angle = compute_angle_degrees(shldr_xy, elbow_xy, wrist_xy)
                    states[side].update(elbow_angle, cfg)
                    color = states[side].last_color
                    # Draw joints and connections
                    cv2.circle(frame, shldr_xy, 7, color, -1)
                    cv2.circle(frame, elbow_xy, 9, color, -1)
                    cv2.circle(frame, wrist_xy, 7, color, -1)
                    cv2.line(frame, shldr_xy, elbow_xy, color, 3)
                    cv2.line(frame, elbow_xy, wrist_xy, color, 3)
                    # Angle text and feedback
                    if states[side].ema_angle is not None:
                        drawer.put_text(frame, f"{int(states[side].ema_angle)}°", (elbow_xy[0] + 10, elbow_xy[1] - 12), color)
                    drawer.put_text(frame, states[side].last_feedback, (elbow_xy[0] + 10, elbow_xy[1] + 20), color)
                    # Progress bar
                    bar_w, bar_h = 18, int(frame_h * 0.6)
                    margin = 16
                    x = margin if side == 'left' else (frame_w - margin - bar_w)
                    y = int((frame_h - bar_h) / 2)
                    drawer.draw_progress_bar(frame, x, y, bar_w, bar_h, states[side].progress_01, color)
                    # Rep count
                    drawer.draw_rep_counter(frame, 10, label_y + 20, states[side].rep_count, side, color)

                # Skeleton
                if draw_cfg.show_pose_skeleton:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                drawer.put_text(frame, 'No person detected', (10, 40), (0, 165, 255))

            # Optional unified overlay
            try:
                from .unified_overlay import draw_header, draw_feedback_panel, draw_fps
            except Exception:
                try:
                    from unified_overlay import draw_header, draw_feedback_panel, draw_fps
                except Exception:
                    draw_header = draw_feedback_panel = draw_fps = None

            if draw_header and draw_fps:
                draw_header(frame, frame_w, 'Tricep Extensions')
                cur_time = time.time()
                fps = 1.0 / max(cur_time - prev_time, 1e-6)
                prev_time = cur_time
                draw_fps(frame, fps)
                # Feedback panel
                full_extension = any(s.last_feedback == 'Full extension' for s in states.values())
                items = [('Full Extension', full_extension)]
                draw_feedback_panel(frame, items, frame_w, frame_h)

            if draw_cfg.show_debug:
                debug_text = f"Top<{int(cfg.top_angle_deg)}  Bottom>{int(cfg.bottom_angle_deg)}  Hold:{cfg.hold_frames_required}"
                cv2.putText(frame, debug_text, (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # Show window
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def build_arg_parser():
    p = argparse.ArgumentParser(description='Tricep Extensions Tracker using MediaPipe Pose')
    p.add_argument('--input', type=str, default='0')
    p.add_argument('--width', type=int, default=0)
    p.add_argument('--height', type=int, default=0)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2])
    p.add_argument('--min-detection-confidence', type=float, default=0.5)
    p.add_argument('--min-tracking-confidence', type=float, default=0.5)
    p.add_argument('--top-angle', type=float, default=30.0)
    p.add_argument('--bottom-angle', type=float, default=110.0)
    p.add_argument('--hold-frames', type=int, default=3)
    p.add_argument('--smoothing-alpha', type=float, default=0.2)
    p.add_argument('--min-visibility', type=float, default=0.6)
    p.add_argument('--no-skeleton', action='store_true')
    p.add_argument('--no-mirror', action='store_true')
    p.add_argument('--debug', action='store_true')
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
