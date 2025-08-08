import argparse
import time
import threading
import queue
import csv
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class ArmSide(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass
class RepCounterConfig:
    lower_angle_threshold: float = 100.0  # angle considered fully contracted
    upper_angle_threshold: float = 160.0  # angle considered fully extended
    smoothing_window_size: int = 7
    min_visibility: float = 0.6
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6


class AngleSmoother:
    def __init__(self, window_size: int = 7) -> None:
        self.values: Deque[float] = deque(maxlen=max(1, window_size))

    def add(self, value: float) -> float:
        self.values.append(value)
        return self.average

    @property
    def average(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))


class TTSAnnouncer:
    def __init__(self, enabled: bool = True, rate: int = 180) -> None:
        self.enabled = enabled
        self._messages: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(rate,), daemon=True)
        self._thread.start()

    def _run(self, rate: int) -> None:
        try:
            import pyttsx3
        except Exception:
            # If TTS library is unavailable, disable quietly
            self.enabled = False
            return

        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", rate)
        except Exception:
            pass

        while not self._stop.is_set():
            try:
                msg = self._messages.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                engine.say(msg)
                engine.runAndWait()
            except Exception:
                # On any TTS failure, disable to avoid spamming errors
                self.enabled = False

        try:
            engine.stop()
        except Exception:
            pass

    def announce(self, text: str) -> None:
        if self.enabled:
            self._messages.put(text)

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Calculate the angle ABC (at point B) in degrees."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def get_landmark_xy(landmarks, idx: int, width: int, height: int) -> Tuple[float, float, float]:
    lm = landmarks[idx]
    return lm.x * width, lm.y * height, lm.visibility


def choose_side_by_visibility(landmarks, width: int, height: int, min_visibility: float) -> Optional[ArmSide]:
    mp_pose = mp.solutions.pose
    left_ids = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
    ]
    right_ids = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
    ]

    def visibility_score(indices: List[int]) -> float:
        vis = 0.0
        for i in indices:
            _, _, v = get_landmark_xy(landmarks, i, width, height)
            vis += 1.0 if v >= min_visibility else 0.0
        return vis

    left_vis = visibility_score(left_ids)
    right_vis = visibility_score(right_ids)

    if left_vis == 0 and right_vis == 0:
        return None
    return ArmSide.LEFT if left_vis >= right_vis else ArmSide.RIGHT


@dataclass
class RepState:
    reps: int = 0
    stage: Optional[str] = None  # "up" or "down"


class RepCounter:
    def __init__(self, config: RepCounterConfig) -> None:
        self.config = config
        self.state = RepState()

    def update(self, angle: float) -> Optional[int]:
        """Update state machine based on current angle.

        Returns the new rep count only when a rep is completed; otherwise None.
        """
        cfg = self.config
        st = self.state

        if angle > cfg.upper_angle_threshold:
            st.stage = "up"
        elif angle < cfg.lower_angle_threshold and st.stage == "up":
            st.stage = "down"
            st.reps += 1
            return st.reps
        return None


def draw_overlay(
    image: np.ndarray,
    angle: float,
    smoothed_angle: float,
    reps: int,
    stage: Optional[str],
    side: Optional[ArmSide],
    fps: float,
) -> None:
    h, w = image.shape[:2]

    # Info panel
    cv2.rectangle(image, (10, 10), (int(w * 0.42), 140), (0, 0, 0), -1)
    cv2.addWeighted(image, 0.6, image, 0, 0, image)

    cv2.putText(image, f"Reps: {reps}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.putText(image, f"Stage: {stage or '-'}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f"Angle: {int(smoothed_angle)} ({int(angle)})", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Side and FPS at top-right
    side_text = side.value if side else "-"
    cv2.putText(image, f"Side: {side_text}", (w - 240, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(image, f"FPS: {fps:.1f}", (w - 240, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 0), 2)


def draw_angle_label(image: np.ndarray, position: Tuple[int, int], angle: float) -> None:
    cv2.putText(
        image,
        str(int(angle)),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def compute_arm_angle(
    landmarks,
    side: ArmSide,
    width: int,
    height: int,
) -> Tuple[float, Tuple[int, int]]:
    mp_pose = mp.solutions.pose
    if side == ArmSide.LEFT:
        shoulder_id = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        elbow_id = mp_pose.PoseLandmark.LEFT_ELBOW.value
        wrist_id = mp_pose.PoseLandmark.LEFT_WRIST.value
    else:
        shoulder_id = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        elbow_id = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        wrist_id = mp_pose.PoseLandmark.RIGHT_WRIST.value

    shoulder_x, shoulder_y, _ = get_landmark_xy(landmarks, shoulder_id, width, height)
    elbow_x, elbow_y, _ = get_landmark_xy(landmarks, elbow_id, width, height)
    wrist_x, wrist_y, _ = get_landmark_xy(landmarks, wrist_id, width, height)

    angle = calculate_angle((shoulder_x, shoulder_y), (elbow_x, elbow_y), (wrist_x, wrist_y))
    return angle, (int(elbow_x), int(elbow_y))


def main():
    parser = argparse.ArgumentParser(description="Upgraded Pose Rep Counter")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--upper-threshold", type=float, default=160.0, help="Angle threshold for fully extended (up) stage")
    parser.add_argument("--lower-threshold", type=float, default=100.0, help="Angle threshold for fully contracted (down) stage")
    parser.add_argument("--smooth-window", type=int, default=7, help="Smoothing window size for angle moving average")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1, help="MediaPipe Pose model complexity")
    parser.add_argument("--min-det", type=float, default=0.6, help="Min detection confidence")
    parser.add_argument("--min-track", type=float, default=0.6, help="Min tracking confidence")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS announcements")
    parser.add_argument("--window", type=str, default="Pose Rep Counter", help="Window title")
    parser.add_argument("--log-csv", type=str, default="", help="Optional CSV path to log angle/reps stream (toggle in-app with 'l')")

    args = parser.parse_args()

    config = RepCounterConfig(
        lower_angle_threshold=args.lower_threshold,
        upper_angle_threshold=args.upper_threshold,
        smoothing_window_size=args.smooth_window,
        min_visibility=0.6,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track,
    )

    cap = cv2.VideoCapture(args.camera)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("ERROR: Could not open camera index", args.camera)
        return

    smoother = AngleSmoother(window_size=config.smoothing_window_size)
    counter = RepCounter(config)
    tts = TTSAnnouncer(enabled=(not args.no_tts))

    # CSV logging
    logging_enabled = bool(args.log_csv)
    csv_path = args.log_csv
    csv_file = None
    csv_writer = None
    if logging_enabled:
        try:
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["ts", "angle_raw", "angle_smoothed", "reps", "stage", "side"])
            print(f"Logging to {csv_path} (toggle with 'l')")
        except Exception as e:
            print(f"WARNING: Could not open CSV for logging: {e}")
            logging_enabled = False

    last_time = time.time()
    fps = 0.0

    mp_drawing = mp.solutions.drawing_utils
    mp_styles = getattr(mp.solutions, "drawing_styles", None)
    mp_pose = mp.solutions.pose

    side_selected: Optional[ArmSide] = None

    with mp_pose.Pose(
        model_complexity=config.model_complexity,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    ) as pose:
        cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("WARNING: Empty frame from camera")
                continue

            frame_height, frame_width = frame.shape[:2]

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            # Back to BGR for display
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            angle_raw = 0.0
            elbow_pos = (int(frame_width * 0.5), int(frame_height * 0.5))
            current_side: Optional[ArmSide] = side_selected

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                if current_side is None:
                    current_side = choose_side_by_visibility(
                        landmarks, frame_width, frame_height, config.min_visibility
                    )

                if current_side is not None:
                    try:
                        angle_raw, elbow_pos = compute_arm_angle(
                            landmarks, current_side, frame_width, frame_height
                        )
                    except Exception:
                        pass

                # Draw pose
                if mp_styles:
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                    )
                else:
                    mp_drawing.draw_landmarks(
                        image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )

            # Update smoothing and counter
            angle_smoothed = smoother.add(angle_raw)
            new_rep = counter.update(angle_smoothed)
            if new_rep is not None:
                tts.announce(f"Rep {new_rep}")

            # Update FPS
            now = time.time()
            dt = max(1e-6, now - last_time)
            fps = (fps * 0.9) + (0.1 * (1.0 / dt))  # smooth FPS
            last_time = now

            # Draw overlays
            draw_angle_label(image_bgr, elbow_pos, angle_smoothed)
            draw_overlay(
                image=image_bgr,
                angle=angle_raw,
                smoothed_angle=angle_smoothed,
                reps=counter.state.reps,
                stage=counter.state.stage,
                side=current_side,
                fps=fps,
            )

            # CSV logging
            if logging_enabled and csv_writer is not None:
                try:
                    csv_writer.writerow(
                        [
                            f"{now:.4f}",
                            f"{angle_raw:.2f}",
                            f"{angle_smoothed:.2f}",
                            counter.state.reps,
                            counter.state.stage or "",
                            (current_side.value if current_side else ""),
                        ]
                    )
                except Exception:
                    pass

            # Help bar
            help_text = "[q] quit  [r] reset  [s] switch side  [t] toggle TTS  [l] toggle logging"
            cv2.rectangle(
                image_bgr,
                (0, frame_height - 30),
                (frame_width, frame_height),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                image_bgr,
                help_text,
                (10, frame_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(args.window, image_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                counter.state = RepState()
                smoother = AngleSmoother(window_size=config.smoothing_window_size)
            elif key == ord("s"):
                if current_side == ArmSide.LEFT:
                    side_selected = ArmSide.RIGHT
                elif current_side == ArmSide.RIGHT:
                    side_selected = ArmSide.LEFT
                else:
                    # if none selected yet, default to LEFT
                    side_selected = ArmSide.LEFT
            elif key == ord("t"):
                enabled = tts.toggle()
                status = "ON" if enabled else "OFF"
                print(f"TTS: {status}")
            elif key == ord("l"):
                if logging_enabled:
                    logging_enabled = False
                    if csv_file:
                        try:
                            csv_file.flush()
                            csv_file.close()
                        except Exception:
                            pass
                        csv_file = None
                        csv_writer = None
                    print("Logging: OFF")
                else:
                    try:
                        csv_path = csv_path or "pose_log.csv"
                        csv_file = open(csv_path, "w", newline="")
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(["ts", "angle_raw", "angle_smoothed", "reps", "stage", "side"])
                        logging_enabled = True
                        print(f"Logging to {csv_path}")
                    except Exception as e:
                        print(f"Could not enable logging: {e}")

    cap.release()
    cv2.destroyAllWindows()
    tts.stop()


if __name__ == "__main__":
    main()