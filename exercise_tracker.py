import streamlit as st
from database import FitnessDatabase, ExerciseSession, ExerciseRep
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
import json

class ExerciseTracker:
    def __init__(self):
        self.db = FitnessDatabase()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize exercise tracking session state"""
        if "current_session" not in st.session_state:
            st.session_state.current_session = None
        if "session_start_time" not in st.session_state:
            st.session_state.session_start_time = None
        if "current_exercise" not in st.session_state:
            st.session_state.current_exercise = None
        if "rep_count" not in st.session_state:
            st.session_state.rep_count = 0
        if "set_count" not in st.session_state:
            st.session_state.set_count = 0
        if "form_scores" not in st.session_state:
            st.session_state.form_scores = []
        if "rep_data" not in st.session_state:
            st.session_state.rep_data = []
    
    def start_exercise_session(self, exercise_type: str, user_id: int):
        """Start a new exercise session"""
        st.session_state.current_exercise = exercise_type
        st.session_state.session_start_time = datetime.now()
        st.session_state.rep_count = 0
        st.session_state.set_count = 0
        st.session_state.form_scores = []
        st.session_state.rep_data = []
        
        st.session_state.current_session = {
            "exercise_type": exercise_type,
            "user_id": user_id,
            "start_time": st.session_state.session_start_time,
            "reps": [],
            "sets": []
        }
        
        st.success(f"Started {exercise_type} session!")
    
    def end_exercise_session(self) -> Optional[Dict]:
        """End the current exercise session and save to database"""
        if not st.session_state.current_session:
            return None
        
        end_time = datetime.now()
        duration = (end_time - st.session_state.session_start_time).total_seconds() / 60
        
        # Calculate session metrics
        total_reps = st.session_state.rep_count
        total_sets = st.session_state.set_count
        avg_form_score = sum(st.session_state.form_scores) / len(st.session_state.form_scores) if st.session_state.form_scores else 0
        
        # Estimate calories burned (simple formula)
        calories_burned = self._estimate_calories(duration, avg_form_score, total_reps)
        
        # Save session to database
        try:
            session_id = self.db.add_exercise_session(
                user_id=st.session_state.current_session["user_id"],
                exercise_type=st.session_state.current_exercise,
                start_time=st.session_state.session_start_time,
                end_time=end_time,
                total_reps=total_reps,
                total_sets=total_sets,
                form_score=round(avg_form_score, 2),
                calories_burned=calories_burned,
                duration_minutes=round(duration, 2)
            )
            
            # Save individual rep data
            for rep_data in st.session_state.rep_data:
                self.db.add_exercise_rep(
                    session_id=session_id,
                    rep_number=rep_data["rep_number"],
                    set_number=rep_data["set_number"],
                    form_score=rep_data["form_score"],
                    rep_time=rep_data["rep_time"],
                    angles=rep_data["angles"],
                    feedback=rep_data["feedback"]
                )
            
            session_summary = {
                "session_id": session_id,
                "exercise_type": st.session_state.current_exercise,
                "duration_minutes": round(duration, 2),
                "total_reps": total_reps,
                "total_sets": total_sets,
                "avg_form_score": round(avg_form_score, 2),
                "calories_burned": calories_burned
            }
            
            # Reset session state
            self._reset_session_state()
            
            return session_summary
            
        except Exception as e:
            st.error(f"Error saving session: {e}")
            return None
    
    def _reset_session_state(self):
        """Reset all session-related state variables"""
        st.session_state.current_session = None
        st.session_state.session_start_time = None
        st.session_state.current_exercise = None
        st.session_state.rep_count = 0
        st.session_state.set_count = 0
        st.session_state.form_scores = []
        st.session_state.rep_data = []
    
    def add_rep(self, form_score: float, rep_time: float, angles: Dict[str, float], 
                feedback: List[str]):
        """Add a completed rep to the current session"""
        if not st.session_state.current_session:
            return
        
        st.session_state.rep_count += 1
        
        # Determine set number (every 10 reps = new set)
        if st.session_state.rep_count % 10 == 1:
            st.session_state.set_count += 1
        
        # Store rep data
        rep_data = {
            "rep_number": st.session_state.rep_count,
            "set_number": st.session_state.set_count,
            "form_score": form_score,
            "rep_time": rep_time,
            "angles": angles,
            "feedback": feedback
        }
        
        st.session_state.rep_data.append(rep_data)
        st.session_state.form_scores.append(form_score)
        
        st.success(f"Rep {st.session_state.rep_count} completed! Form score: {form_score:.1f}%")
    
    def _estimate_calories(self, duration_minutes: float, form_score: float, total_reps: int) -> float:
        """Estimate calories burned during exercise session"""
        # Base calories per minute for strength training
        base_calories_per_minute = 4.0
        
        # Adjust based on form score (better form = more effective workout)
        form_multiplier = 0.8 + (form_score / 100) * 0.4
        
        # Adjust based on intensity (more reps = higher intensity)
        intensity_multiplier = 1.0 + (total_reps / 100) * 0.3
        
        calories = duration_minutes * base_calories_per_minute * form_multiplier * intensity_multiplier
        
        return round(calories, 1)
    
    def get_session_progress(self) -> Dict:
        """Get current session progress"""
        if not st.session_state.current_session:
            return {}
        
        duration = 0
        if st.session_state.session_start_time:
            duration = (datetime.now() - st.session_state.session_start_time).total_seconds() / 60
        
        return {
            "exercise_type": st.session_state.current_exercise,
            "duration_minutes": round(duration, 2),
            "rep_count": st.session_state.rep_count,
            "set_count": st.session_state.set_count,
            "avg_form_score": sum(st.session_state.form_scores) / len(st.session_state.form_scores) if st.session_state.form_scores else 0
        }

def render_exercise_tracker():
    """Render the exercise tracking interface"""
    st.markdown("## 🏋️ Exercise Tracker")
    
    tracker = ExerciseTracker()
    
    # Check if user is logged in
    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.error("Please log in to use the exercise tracker")
        return
    
    # Session controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not tracker.get_session_progress():
            exercise_type = st.selectbox("Select Exercise", [
                "Bicep Curls", "Squats", "Shoulder Press", "Lateral Raise"
            ])
            
            if st.button("🎬 Start Session", type="primary"):
                tracker.start_exercise_session(exercise_type, st.session_state.user_id)
                st.rerun()
    
    with col2:
        if tracker.get_session_progress():
            if st.button("⏹️ End Session", type="secondary"):
                session_summary = tracker.end_exercise_session()
                if session_summary:
                    st.success("Session completed and saved!")
                    st.rerun()
    
    with col3:
        if tracker.get_session_progress():
            if st.button("🔄 Reset Session", type="secondary"):
                tracker._reset_session_state()
                st.rerun()
    
    # Current session display
    progress = tracker.get_session_progress()
    if progress:
        st.markdown("### 📊 Current Session")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Exercise", progress["exercise_type"])
        with col2:
            st.metric("Duration", f"{progress['duration_minutes']:.1f} min")
        with col3:
            st.metric("Reps", progress["rep_count"])
        with col4:
            st.metric("Sets", progress["set_count"])
        
        # Form score display
        if progress["avg_form_score"] > 0:
            st.markdown(f"**Average Form Score:** {progress['avg_form_score']:.1f}%")
            
            # Form score gauge
            form_score = progress["avg_form_score"]
            if form_score >= 90:
                st.success("🎯 Excellent form! Keep it up!")
            elif form_score >= 80:
                st.info("👍 Good form! Minor improvements possible.")
            elif form_score >= 70:
                st.warning("⚠️ Fair form. Focus on technique.")
            else:
                st.error("❌ Poor form. Consider reducing intensity.")
        
        # Manual rep input
        st.markdown("### ➕ Add Rep Manually")
        col1, col2 = st.columns(2)
        
        with col1:
            form_score = st.slider("Form Score (%)", 0, 100, 80)
            rep_time = st.number_input("Rep Time (seconds)", min_value=0.1, value=2.0, step=0.1)
        
        with col2:
            # Simulated joint angles (in real app, these would come from pose detection)
            angles = {
                "elbow_angle": st.number_input("Elbow Angle (deg)", 0, 180, 90),
                "shoulder_angle": st.number_input("Shoulder Angle (deg)", 0, 180, 90),
                "knee_angle": st.number_input("Knee Angle (deg)", 0, 180, 90)
            }
            
            feedback_options = [
                "Good form", "Elbow drift", "Knee cave", "Poor posture",
                "Too fast", "Too slow", "Range of motion issue"
            ]
            feedback = st.multiselect("Form Feedback", feedback_options)
        
        if st.button("✅ Add Rep"):
            tracker.add_rep(form_score, rep_time, angles, feedback)
            st.rerun()
        
        # Session history
        if st.session_state.rep_data:
            st.markdown("### 📝 Rep History")
            
            # Create a simple table of recent reps
            recent_reps = st.session_state.rep_data[-10:]  # Show last 10 reps
            
            rep_data = []
            for rep in recent_reps:
                rep_data.append({
                    "Rep": rep["rep_number"],
                    "Set": rep["set_number"],
                    "Form Score": f"{rep['form_score']:.1f}%",
                    "Time": f"{rep['rep_time']:.1f}s",
                    "Feedback": ", ".join(rep["feedback"]) if rep["feedback"] else "None"
                })
            
            if rep_data:
                import pandas as pd
                df = pd.DataFrame(rep_data)
                st.dataframe(df, use_container_width=True)
    
    # Exercise tips
    if progress and progress["exercise_type"]:
        st.markdown("### 💡 Exercise Tips")
        
        tips = {
            "Bicep Curls": [
                "Keep your elbows close to your body",
                "Control the movement on both up and down",
                "Avoid swinging or using momentum",
                "Focus on squeezing your biceps at the top"
            ],
            "Squats": [
                "Keep your chest up and back straight",
                "Push your knees out in line with your toes",
                "Go as deep as your mobility allows",
                "Drive through your heels to stand up"
            ],
            "Shoulder Press": [
                "Keep your core tight throughout the movement",
                "Press directly overhead, not forward",
                "Control the weight down to starting position",
                "Avoid arching your back"
            ],
            "Lateral Raise": [
                "Keep a slight bend in your elbows",
                "Raise arms to shoulder height, not higher",
                "Control the movement down",
                "Avoid shrugging your shoulders"
            ]
        }
        
        exercise_tips = tips.get(progress["exercise_type"], [])
        for tip in exercise_tips:
            st.markdown(f"- {tip}")

def render_workout_page():
    """Render the main workout page"""
    st.markdown("## 💪 Workout Session")
    
    # Check if user is logged in
    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.error("Please log in to access workouts")
        return
    
    # Page navigation
    tab1, tab2, tab3 = st.tabs(["🏋️ Exercise Tracker", "📊 Session History", "🎯 Workout Goals"])
    
    with tab1:
        render_exercise_tracker()
    
    with tab2:
        render_session_history()
    
    with tab3:
        render_workout_goals()

def render_session_history():
    """Render session history for the current user"""
    st.markdown("### 📊 Your Session History")
    
    user_manager = st.session_state.get("user_manager", None)
    if not user_manager:
        from user_management import UserManager
        user_manager = UserManager()
        st.session_state.user_manager = user_manager
    
    user = user_manager.get_current_user()
    if not user:
        st.error("User not found")
        return
    
    # Get recent sessions
    progress = user_manager.db.get_user_progress(user.id, days=30)
    
    if progress["total_sessions"] > 0:
        st.markdown(f"**Last 30 Days Summary:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sessions", progress["total_sessions"])
        with col2:
            st.metric("Total Reps", progress["total_reps"])
        with col3:
            st.metric("Avg Form Score", f"{progress['avg_form_score']}%")
        
        # Show progress trend if available
        if progress["progress_trend"]:
            st.markdown("### 📈 Progress Trend")
            
            import pandas as pd
            import plotly.express as px
            
            df = pd.DataFrame(progress["progress_trend"])
            df['week'] = pd.to_datetime(df['week'])
            
            fig = px.line(df, x='week', y=['avg_reps', 'avg_form_score'], 
                         title="Weekly Performance Trends",
                         labels={'value': 'Score', 'variable': 'Metric'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No workout sessions yet. Start your first workout to see your progress!")

def render_workout_goals():
    """Render workout goals and achievements"""
    st.markdown("### 🎯 Workout Goals & Achievements")
    
    user_manager = st.session_state.get("user_manager", None)
    if not user_manager:
        from user_management import UserManager
        user_manager = UserManager()
        st.session_state.user_manager = user_manager
    
    user = user_manager.get_current_user()
    if not user:
        st.error("User not found")
        return
    
    # Get user recommendations
    recommendations = user_manager.db.get_exercise_recommendations(user.id)
    
    if recommendations:
        st.markdown("#### 🚀 Your Next Workout")
        next_workout = recommendations["next_workout"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Level", next_workout["level"].title())
        with col2:
            st.metric("Duration", f"{next_workout['estimated_duration']} min")
        with col3:
            st.metric("Focus", next_workout["focus"].title())
        
        # Show exercise breakdown
        st.markdown("**Exercise Breakdown:**")
        for exercise, details in next_workout["exercises"].items():
            exercise_name = exercise.replace("_", " ").title()
            st.markdown(f"- **{exercise_name}**: {details['sets']} sets × {details['reps']} reps")
        
        # Progression tips
        if recommendations.get("progression_tips"):
            st.markdown("#### 💡 Progression Tips")
            for tip in recommendations["progression_tips"]:
                st.info(tip)
    
    # Weekly goals
    st.markdown("#### 📅 Weekly Goals")
    
    col1, col2 = st.columns(2)
    with col1:
        weekly_sessions = st.number_input("Target Sessions", min_value=1, max_value=7, value=3)
        weekly_reps = st.number_input("Target Reps", min_value=50, max_value=500, value=100, step=10)
    
    with col2:
        weekly_form_score = st.number_input("Target Form Score (%)", min_value=70, max_value=100, value=85)
        weekly_duration = st.number_input("Target Duration (min)", min_value=15, max_value=120, value=30, step=5)
    
    if st.button("🎯 Set Weekly Goals"):
        st.success("Weekly goals updated!")
    
    # Achievement badges (placeholder)
    st.markdown("#### 🏆 Achievements")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🥇 **First Workout**")
        st.markdown("Complete your first exercise session")
    
    with col2:
        st.markdown("🥈 **Form Master**")
        st.markdown("Achieve 90%+ form score")
    
    with col3:
        st.markdown("🥉 **Consistency King**")
        st.markdown("Work out 3+ times per week")
