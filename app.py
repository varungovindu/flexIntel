import os
import sys
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Union

import streamlit as st

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="flexIntel - AI Fitness Coach",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- STYLING -----------------
st.markdown(
    """
    <style>
    :root {
      --primary: #0066cc;
      --primary-dark: #005bb5;
      --bg-soft: #eaf3fb;
      --text-subtle: #4a5568;
    }
    .title { text-align: center; font-size: 2.2em; font-weight: 800; color: var(--primary); }
    .subtitle { text-align: center; font-size: 1.05em; color: var(--text-subtle); }
    .tip-box { background-color: var(--bg-soft); padding: 10px 15px; border-left: 5px solid var(--primary); border-radius: 8px; color: #1f3b57; }

    /* Buttons */
    .stButton>button { background: var(--primary) !important; color: #ffffff !important; padding: 10px 18px; font-size: 16px; border-radius: 8px; border: none; font-weight: 600; }
    .stButton>button:hover { background: var(--primary-dark) !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #ffffff 0%, #f4f8fc 100%); }

    /* Inputs accent */
    input[type="checkbox"], input[type="radio"] { accent-color: var(--primary); }
    div[role="slider"] { color: var(--primary); }

    /* Layout tweaks */
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- HEADER -----------------
st.markdown('<div class="title">🤖 flexIntel - Your AI Fitness Coach</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Train smarter. Move better. Track perfectly.</div><br>', unsafe_allow_html=True)

# ----------------- PATHS -----------------
ROOT_DIR = Path(__file__).resolve().parent
TRACKERS = {
    "💪 Bicep Curls": ROOT_DIR / "bicep_curl_tracker.py",
    "🏋️ Squats": ROOT_DIR / "squat_tracker.py",
    "🏋️‍♂️ Shoulder Press": ROOT_DIR / "shoulder_press_tracker.py",
    "🦾 Lateral Raise": ROOT_DIR / "lateral_raise_tracker.py",
}

if "proc" not in st.session_state:
    st.session_state.proc = None

# ----------------- SIDEBAR -----------------
st.sidebar.header("📋 Select an Exercise")
exercise_label = st.sidebar.selectbox("Choose", list(TRACKERS.keys()))
script_path = TRACKERS[exercise_label]

st.sidebar.markdown("---")
st.sidebar.header("🎥 Video Source")
source_type = st.sidebar.radio("Input Type", ["Webcam", "Video file"], horizontal=True)
input_value = "0"
if source_type == "Webcam":
    cam_index = st.sidebar.number_input("Camera index", min_value=0, value=0, step=1)
    input_value = str(cam_index)
else:
    input_value = st.sidebar.text_input("Video file path", value="sample.mp4")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Capture Settings")
cap_cols = st.sidebar.columns(3)
with cap_cols[0]:
    cap_w = st.number_input("Width", min_value=0, value=0, step=2, help="0=default")
with cap_cols[1]:
    cap_h = st.number_input("Height", min_value=0, value=0, step=2, help="0=default")
with cap_cols[2]:
    cap_fps = st.number_input("FPS", min_value=1, value=30, step=1)

st.sidebar.markdown("---")
st.sidebar.header("🧠 Model Settings")
model_complexity = st.sidebar.select_slider("Model complexity", options=[0, 1, 2], value=1)
min_det_conf = st.sidebar.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
min_track_conf = st.sidebar.slider("Min tracking confidence", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🧩 Display & Debug")
mirror = st.sidebar.checkbox("Mirror (selfie)", value=True)
skeleton = st.sidebar.checkbox("Draw skeleton", value=True)
debug = st.sidebar.checkbox("Debug overlay")
no_window = st.sidebar.checkbox("Run headless (no window)")
output_path = st.sidebar.text_input("Save annotated video (optional)", value="")

# ----------------- EXERCISE-SPECIFIC CONTROLS -----------------
with st.expander("Exercise parameters", expanded=True):
    if exercise_label == "💪 Bicep Curls":
        top_angle = st.slider("Top angle (deg)", 30, 90, 55)
        bottom_angle = st.slider("Bottom angle (deg)", 120, 180, 165)
        smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.2, 0.05)
        hold_frames = st.slider("Hold frames", 1, 10, 3)
        min_visibility = st.slider("Min visibility", 0.0, 1.0, 0.6, 0.05)
        elbow_drift_ratio = st.slider("Elbow drift warn ratio", 0.0, 0.2, 0.07, 0.005)
    elif exercise_label == "🏋️‍♂️ Shoulder Press":
        top_off = st.slider("Top offset ratio (frame_h)", 0.0, 0.2, 0.07, 0.005)
        bottom_off = st.slider("Bottom offset ratio (frame_h)", 0.0, 0.2, 0.08, 0.005)
        hold_frames = st.slider("Hold frames", 1, 10, 3)
        smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.2, 0.05)
        min_visibility = st.slider("Min visibility", 0.0, 1.0, 0.6, 0.05)
        vertical_tol = st.slider("Vertical forearm tolerance (deg)", 0, 60, 25)
        guides = st.checkbox("Show threshold guides")
    elif exercise_label == "🏋️ Squats":
        top_angle = st.slider("Bottom-of-squat knee angle (deg)", 50, 120, 85)
        bottom_angle = st.slider("Standing knee angle (deg)", 150, 180, 170)
        hold_frames = st.slider("Hold frames", 1, 10, 3)
        smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.25, 0.05)
        min_visibility = st.slider("Min visibility", 0.0, 1.0, 0.6, 0.05)
        torso_tol = st.slider("Torso forward tolerance (deg)", 0, 90, 35)
        knee_cave_ratio = st.slider("Knee cave warn ratio", 0.5, 1.0, 0.8, 0.01)
        valgus_warning = st.checkbox("Enable knees-in warning")
    elif exercise_label == "🦾 Lateral Raise":
        top_angle = st.slider("Top angle (deg)", 60, 120, 85)
        bottom_angle = st.slider("Bottom angle (deg)", 0, 60, 25)
        hold_frames = st.slider("Hold frames", 1, 10, 3)
        smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.2, 0.05)
        min_visibility = st.slider("Min visibility", 0.0, 1.0, 0.6, 0.05)
        elbow_lead_check = st.checkbox("Enable elbow-lead check", value=True)
        shrug_warn_ratio = st.slider("Shrug warn ratio (frame_h)", 0.0, 0.1, 0.04, 0.005)

# ----------------- COMMAND BUILDER -----------------

def py_executable() -> str:
    # Prefer current Python executable
    return sys.executable or "python3"


def build_command() -> List[str]:
    args = [py_executable(), str(script_path)]
    # input
    args += ["--input", input_value]
    # capture
    args += ["--fps", str(int(cap_fps))]
    if cap_w:
        args += ["--width", str(int(cap_w))]
    if cap_h:
        args += ["--height", str(int(cap_h))]
    # model
    args += ["--model-complexity", str(int(model_complexity))]
    args += ["--min-detection-confidence", f"{min_det_conf:.2f}"]
    args += ["--min-tracking-confidence", f"{min_track_conf:.2f}"]
    # display
    if not skeleton:
        args += ["--no-skeleton"]
    if not mirror:
        args += ["--no-mirror"]
    if debug:
        args += ["--debug"]
    if no_window:
        args += ["--no-window"]
    if output_path:
        args += ["--output", output_path]

    # exercise-specific
    if exercise_label == "💪 Bicep Curls":
        args += ["--top-angle", str(int(top_angle))]
        args += ["--bottom-angle", str(int(bottom_angle))]
        args += ["--hold-frames", str(int(hold_frames))]
        args += ["--smoothing-alpha", f"{smoothing_alpha:.3f}"]
        args += ["--min-visibility", f"{min_visibility:.2f}"]
        args += ["--elbow-drift-ratio", f"{elbow_drift_ratio:.3f}"]
    elif exercise_label == "🏋️‍♂️ Shoulder Press":
        args += ["--top-offset-ratio", f"{top_off:.3f}"]
        args += ["--bottom-offset-ratio", f"{bottom_off:.3f}"]
        args += ["--hold-frames", str(int(hold_frames))]
        args += ["--smoothing-alpha", f"{smoothing_alpha:.3f}"]
        args += ["--min-visibility", f"{min_visibility:.2f}"]
        args += ["--vertical-forearm-tolerance", str(int(vertical_tol))]
        if guides:
            args += ["--guides"]
    elif exercise_label == "🏋️ Squats":
        args += ["--top-angle", str(int(top_angle))]
        args += ["--bottom-angle", str(int(bottom_angle))]
        args += ["--hold-frames", str(int(hold_frames))]
        args += ["--smoothing-alpha", f"{smoothing_alpha:.3f}"]
        args += ["--min-visibility", f"{min_visibility:.2f}"]
        args += ["--torso-forward-tolerance", str(int(torso_tol))]
        args += ["--knee-cave-ratio", f"{knee_cave_ratio:.2f}"]
        if valgus_warning:
            args += ["--valgus-warning"]
    elif exercise_label == "🦾 Lateral Raise":
        args += ["--top-angle", str(int(top_angle))]
        args += ["--bottom-angle", str(int(bottom_angle))]
        args += ["--hold-frames", str(int(hold_frames))]
        args += ["--smoothing-alpha", f"{smoothing_alpha:.3f}"]
        args += ["--min-visibility", f"{min_visibility:.2f}"]
        if not elbow_lead_check:
            args += ["--no-elbow-lead-check"]
        args += ["--shrug-warn-ratio", f"{shrug_warn_ratio:.3f}"]

    return args


def stop_running_process():
    proc = st.session_state.proc
    if proc is not None and proc.poll() is None:
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
    st.session_state.proc = None


# ----------------- MAIN CONTROLS -----------------
left, right = st.columns([1.2, 1])
with left:
    st.markdown(f"<div class='tip-box'>👉 Tip: Ensure proper lighting and position yourself within the webcam frame.</div>", unsafe_allow_html=True)

with right:
    if st.session_state.proc and st.session_state.proc.poll() is None:
        st.info("🟡 Running… Close the camera window or press Stop.")
        if st.button("⏹️ Stop"):
            stop_running_process()
    else:
        if st.button("🎥 Start Exercise", type="primary"):
            if not script_path.exists():
                st.error("❌ Exercise logic file not found!")
            else:
                cmd = build_command()
                try:
                    preexec = None
                    if os.name != "nt":
                        preexec = os.setsid
                    st.session_state.proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        preexec_fn=preexec,
                    )
                    st.success("✅ Launched. A camera window should appear.")
                except Exception as e:
                    st.error(f"Failed to start: {e}")

# Show command preview and (optional) live logs
with st.expander("Command & Logs", expanded=False):
    cmd_preview = " ".join(shlex.quote(p) for p in build_command())
    st.markdown("**Command**:")
    st.code(cmd_preview, language="bash")

    proc = st.session_state.proc
    if proc and proc.poll() is None:
        st.markdown("**Live output**:")
        # Read a small chunk to avoid blocking UI; not perfect but useful
        try:
            lines: List[str] = []
            for _ in range(50):
                if proc.poll() is not None:
                    break
                line = proc.stdout.readline()
                if not line:
                    break
                lines.append(line.rstrip())
            if lines:
                st.text("\n".join(lines))
        except Exception:
            st.caption("(logging unavailable)")

st.markdown("---")
st.caption("Note: Real-time video opens in a separate window. Close it (or press Stop) to return to Streamlit.")