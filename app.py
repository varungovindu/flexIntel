import os
import sys
import signal
import subprocess
from pathlib import Path
from typing import Dict, List

import streamlit as st

from user_management import (
    UserManager,
    render_login_page,
    render_profile_creation,
    render_user_dashboard,
    render_progress_page,
    render_workout_plans_page,
    render_settings_page,
)
from exercise_tracker import render_workout_page
from coach_chat import render_coach_chat_page
from nutrition import render_nutrition_page


def _set_page_config() -> None:
    st.set_page_config(
        page_title="flexIntel",
        page_icon="🏋️‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --secondary: #10b981;
        --accent: #f97316;
        --bg-soft: #f8fafc;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --border: #e2e8f0;
        --shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
        }

        .main-title { margin: 0; color: var(--primary); }
        .main-subtitle { margin: 0; color: var(--text-secondary); }
        .exercise-title { margin: 0; color: var(--text-primary); }
        .exercise-subtitle { font-size: 1rem; margin: 0.5rem 0 0 0; opacity: 0.9; }
        .control-section { background: var(--bg-soft); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; border: 1px solid var(--border); }
        .control-title { font-size: 1.2rem; font-weight: 600; color: var(--text-primary); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
        .stButton>button { background: var(--primary) !important; color: #fff !important; padding: 12px 24px !important; font-size: 16px !important; border-radius: 8px !important; border: none !important; font-weight: 600 !important; transition: all 0.3s ease !important; box-shadow: var(--shadow) !important; }
        .stButton>button:hover { background: var(--primary-dark) !important; transform: translateY(-2px) !important; box-shadow: 0 6px 12px rgba(0,102,204,0.3) !important; }
        .status-running { background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: white; padding: 1rem; border-radius: 8px; text-align: center; font-weight: 600; margin: 1rem 0; }
        .status-success { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 1rem; border-radius: 8px; text-align: center; font-weight: 600; margin: 1rem 0; }
        .status-error { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; padding: 1rem; border-radius: 8px; text-align: center; font-weight: 600; margin: 1rem 0; }
        .info-box { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; text-align: center; }
        .block-container { padding-top: 1rem; max-width: 1400px; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #ffffff 0%, var(--bg-soft) 100%); border-right: 1px solid var(--border); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_user_manager() -> UserManager:
    if "user_manager" not in st.session_state:
        st.session_state.user_manager = UserManager()
    return st.session_state.user_manager


def _build_command(
    script_path: Path,
    input_value: str,
    fps: int,
    model_complexity: int,
    min_det_conf: float,
    **kwargs,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_value),
        "--fps",
        str(int(fps)),
        "--model-complexity",
        str(int(model_complexity)),
        "--min-detection-confidence",
        str(float(min_det_conf)),
    ]
    flag_map: Dict[str, str] = {
        "top_angle": "--top-angle",
        "bottom_angle": "--bottom-angle",
        "smoothing_alpha": "--smoothing-alpha",
        "top_off": "--top-offset-ratio",
        "bottom_off": "--bottom-offset-ratio",
        "min_hold_time": "--min-hold-time",
        "max_hip_drift": "--max-hip-drift",
        "width": "--width",
        "height": "--height",
    }
    for key, value in kwargs.items():
        flag = flag_map.get(key)
        if flag is not None and value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def _stop_running_process() -> None:
    proc = st.session_state.get("proc")
    if not proc:
        return
    try:
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        st.session_state.proc = None


def _render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1 class="main-title">🤖 flexIntel</h1>
            <p class="main-subtitle">Your AI-Powered Fitness Coach</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(user_manager: UserManager) -> None:
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: var(--primary); margin: 0;">🧭 Navigation</h2>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Navigate through different sections</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not user_manager.is_logged_in():
        st.sidebar.markdown(
            """
            <div class="control-section">
                <h3 style="color: var(--primary); margin-bottom: 1rem;">🔐 Account</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🔑 Login", key="sidebar_login", use_container_width=True):
            st.session_state.current_page = "login"
            st.rerun()
        if st.sidebar.button("📝 Create Profile", key="sidebar_register", use_container_width=True):
            st.session_state.current_page = "register"
            st.rerun()
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            """
            <div class="info-box">
                <h3>🚀 Get Started</h3>
                <p>Create an account to access all features</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.sidebar.markdown(
        """
        <div class="control-section">
            <h3 style="color: var(--primary); margin-bottom: 1rem;">👤 Welcome Back!</h3>
        <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">Ready for your next workout?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for title, page, description in [
        ("📊 Dashboard", "dashboard", "View your fitness overview"),
        ("🏋️ Workout", "workout", "Track your exercises"),
        ("📈 Progress", "progress", "Analyze your performance"),
        ("📋 Plans", "plans", "Your workout plans"),
        ("💬 Coach Chat", "coach", "Diet plans and workout suggestions"),
        ("🍽️ Diet & Calories", "nutrition", "Calorie calculator and diet plan"),
        ("⚙️ Settings", "settings", "Manage your profile"),
    ]:
        if st.sidebar.button(title, key=f"nav_{page}", use_container_width=True, help=description):
            st.session_state.current_page = page
            st.rerun()
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", key="sidebar_logout", use_container_width=True):
        user_manager.logout()
        st.rerun()
    user_info = user_manager.get_current_user()
    if user_info:
        st.sidebar.markdown(
            f"""
        <div class=\"control-section\">
        <h4 style=\"color: var(--text-primary); margin-bottom: 0.5rem;\">👤 {user_info.username}</h4>
        <p style=\"color: var(--text-secondary); margin: 0; font-size: 0.8rem;\">Level: {user_info.fitness_level}<br>Goal: {user_info.fitness_goal}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def _render_footer() -> None:
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0; background: var(--bg-soft); border-radius: 12px; margin: 2rem 0;">
            <h3 style="color: var(--primary); margin-bottom: 0.5rem;">🏋️‍♂️ flexIntel</h3>
        <p style="color: var(--text-secondary); margin: 0; font-size: 1rem;">AI-Powered Fitness Coaching • Track your progress • Improve your form • Achieve your fitness goals</p>
            <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 1rem;">
                <span style="color: var(--text-secondary); font-size: 0.9rem;">🎯 Smart Tracking</span>
                <span style="color: var(--text-secondary); font-size: 0.9rem;">🧠 Adaptive AI</span>
                <span style="color: var(--text-secondary); font-size: 0.9rem;">📱 Real-time Feedback</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_legacy_integration(user_manager: UserManager) -> None:
    if not user_manager.is_logged_in():
        return
    st.markdown("---")
    st.markdown(
        """
        <div class="exercise-container">
            <div class="exercise-header">
                <h1 class="exercise-title">🔗 Legacy Exercise Integration</h1>
                <p class="exercise-subtitle">Access the original pose detection exercises with enhanced styling</p>
            </div>
        """,
        unsafe_allow_html=True,
    )
    root_dir = Path(__file__).resolve().parent / "exercises"
    trackers: Dict[str, Path] = {
        "💪 Tricep Extensions": root_dir / "tricep_extensions.py",
        "🏋️ Squats": root_dir / "squat.py",
        "🏋️‍♂️ Shoulder Press": root_dir / "shoulder_press.py",
        "🦾 Lateral Raise": root_dir / "lateral_raise.py",
        "🦵 Pushups": root_dir / "pushup_tracker.py",
        "🧘 Plank": root_dir / "plank_tracker.py",
    }
    st.markdown(
        """
    <div class="control-section"><h3 class="control-title">🎯 Choose Your Exercise</h3></div>
    """,
        unsafe_allow_html=True,
    )
    exercise_label = st.selectbox("Select Exercise Type", list(trackers.keys()), help="Choose the exercise you want to perform")
    script_path = trackers[exercise_label]
    st.markdown(
        """
    <div class="control-section"><h3 class="control-title">📹 Video Configuration</h3></div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        source_type = st.radio("Input Type", ["Webcam", "Video file"], horizontal=True)
        input_value = "0"
        if source_type == "Webcam":
            cam_index = st.number_input("Camera index", min_value=0, value=0, step=1)
            input_value = str(cam_index)
        else:
            input_value = st.text_input("Video file path", value="sample.mp4")
    with col2:
        cap_fps = st.number_input("FPS", min_value=1, value=30, step=1)
    cap_width = st.number_input("Width", min_value=0, value=0, step=80)
    cap_height = st.number_input("Height", min_value=0, value=0, step=80)
    model_complexity = st.select_slider("Model complexity", options=[0, 1, 2], value=1)
    min_det_conf = st.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
    st.markdown(
        """
    <div class="control-section"><h3 class="control-title">⚙️ Exercise Parameters</h3></div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Configure Exercise Parameters", expanded=False):
        if exercise_label == "💪 Tricep Extensions":
            col1, col2, col3 = st.columns(3)
            with col1:
                top_angle = st.slider("Top angle (deg)", 30, 90, 55)
            with col2:
                bottom_angle = st.slider("Bottom angle (deg)", 120, 180, 165)
            with col3:
                smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.2, 0.05)
        elif exercise_label == "🏋️‍♂️ Shoulder Press":
            col1, col2, col3 = st.columns(3)
            with col1:
                top_off = st.slider("Top offset ratio", 0.0, 0.2, 0.07, 0.005)
            with col2:
                bottom_off = st.slider("Bottom offset ratio", 0.0, 0.2, 0.08, 0.005)
            with col3:
                smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.2, 0.05)
        elif exercise_label == "🏋️ Squats":
            col1, col2, col3 = st.columns(3)
            with col1:
                top_angle = st.slider("Bottom-of-squat knee angle (deg)", 50, 120, 85)
            with col2:
                bottom_angle = st.slider("Standing knee angle (deg)", 150, 180, 170)
            with col3:
                smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.25, 0.05)
        elif exercise_label == "🦾 Lateral Raise":
            col1, col2, col3 = st.columns(3)
            with col1:
                top_angle = st.slider("Top angle (deg)", 60, 120, 85)
            with col2:
                bottom_angle = st.slider("Bottom angle (deg)", 0, 60, 25)
            with col3:
                smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.2, 0.05)
        elif exercise_label == "🦵 Pushups":
            col1, col2, col3 = st.columns(3)
            with col1:
                top_angle = st.slider("Top angle (deg)", 150, 180, 170)
            with col2:
                bottom_angle = st.slider("Bottom angle (deg)", 60, 120, 90)
            with col3:
                smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.2, 0.05)
        elif exercise_label == "🧘 Plank":
            col1, col2, col3 = st.columns(3)
            with col1:
                min_hold_time = st.slider("Min hold time (s)", 5.0, 30.0, 10.0, 0.5)
            with col2:
                max_hip_drift = st.slider("Max hip drift", 0.05, 0.3, 0.15, 0.01)
            with col3:
                smoothing_alpha = st.slider("Smoothing alpha", 0.0, 1.0, 0.3, 0.05)
    st.markdown(
        """
    <div class="control-section"><h3 class="control-title">🎮 Exercise Controls</h3></div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        if "proc" not in st.session_state:
            st.session_state.proc = None
        if st.session_state.proc and st.session_state.proc.poll() is None:
            st.markdown(
                """
    <div class="status-running">🟡 Exercise is running... Close the camera window or press Stop to end</div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("⏹️ Stop Exercise", key="stop_exercise", use_container_width=True):
                _stop_running_process()
                st.success("Exercise stopped successfully!")
                st.rerun()
        else:
            if st.button("🎥 Start Exercise", key="start_exercise", use_container_width=True, type="primary"):
                if not script_path.exists():
                    st.markdown(
                        """
    <div class="status-error">❌ Exercise logic file not found!</div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    kwargs: Dict[str, float] = {}
                    if exercise_label in {"💪 Tricep Extensions", "🏋️ Squats", "🦾 Lateral Raise", "🦵 Pushups"}:
                        kwargs.update({"top_angle": top_angle, "bottom_angle": bottom_angle, "smoothing_alpha": smoothing_alpha})
                    elif exercise_label == "🏋️‍♂️ Shoulder Press":
                        kwargs.update({"top_off": top_off, "bottom_off": bottom_off, "smoothing_alpha": smoothing_alpha})
                    elif exercise_label == "🧘 Plank":
                        kwargs.update({"min_hold_time": min_hold_time, "max_hip_drift": max_hip_drift, "smoothing_alpha": smoothing_alpha})
                    cmd = _build_command(
                        script_path,
                        input_value,
                        cap_fps,
                        model_complexity,
                        min_det_conf,
                        width=int(cap_width),
                        height=int(cap_height),
                        **kwargs,
                    )
                    try:
                        preexec = None
                        creationflags = 0
                        if os.name != "nt":
                            preexec = os.setsid
                        else:
                            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        st.session_state.proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            universal_newlines=True,
                            preexec_fn=preexec,
                            creationflags=creationflags,
                        )
                        st.markdown(
                            """
    <div class="status-success">✅ Exercise launched successfully! A camera window should appear.</div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.markdown(
                            f"""
    <div class=\"status-error\">❌ Failed to start exercise: {e}</div>
                            """,
                            unsafe_allow_html=True,
                        )
    with col2:
        st.markdown(
            """
    <div class="info-box"><h3>📱 Integration Note</h3><p>This legacy integration runs the original pose detection exercises. For a better experience with progress tracking and adaptive workouts, use the <strong>Workout</strong> tab above!</p></div>
    """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _set_page_config()
    _inject_css()
    user_manager = _get_user_manager()
    _render_header()
    _render_sidebar(user_manager)
    if not user_manager.is_logged_in():
        current_page = st.session_state.get("current_page", "login")
        if current_page == "login":
            render_login_page()
        elif current_page == "register":
            render_profile_creation()
        _render_footer()
        return
    current_page = st.session_state.get("current_page", "dashboard")
    page_titles: Dict[str, List[str]] = {
        "dashboard": ["📊 Dashboard", "Your fitness overview and quick stats"],
        "workout": ["🏋️ Workout", "Track exercises and monitor form"],
        "progress": ["📈 Progress", "Analyze your performance trends"],
        "plans": ["📋 Workout Plans", "Your personalized workout recommendations"],
        "coach": ["💬 Coach Chat", "Ask for diet or training guidance"],
        "nutrition": ["🍽️ Diet & Calories", "Calorie calculator and diet plan"],
        "settings": ["⚙️ Settings", "Manage your profile and preferences"],
    }
    if current_page in page_titles:
        title, subtitle = page_titles[current_page]
        st.markdown(
            f"""
    <div class=\"exercise-header\"><h1 class=\"exercise-title\">{title}</h1><p class=\"exercise-subtitle\">{subtitle}</p></div>
    """,
            unsafe_allow_html=True,
        )
    if current_page == "dashboard":
        render_user_dashboard()
    elif current_page == "workout":
        render_workout_page()
    elif current_page == "progress":
        render_progress_page()
    elif current_page == "plans":
        render_workout_plans_page()
    elif current_page == "coach":
        render_coach_chat_page()
    elif current_page == "nutrition":
        render_nutrition_page()
    elif current_page == "settings":
        render_settings_page()
    _render_footer()
    _render_legacy_integration(user_manager)


if __name__ == "__main__":
    main()