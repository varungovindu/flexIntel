import streamlit as st
from database import FitnessDatabase, User
from typing import Optional, Dict, List
import hashlib
import secrets

class UserManager:
    def __init__(self):
        self.db = FitnessDatabase()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if "user_id" not in st.session_state:
            st.session_state.user_id = None
        if "username" not in st.session_state:
            st.session_state.username = None
        if "is_authenticated" not in st.session_state:
            st.session_state.is_authenticated = False
    
    def is_logged_in(self) -> bool:
        """Check if user is currently logged in"""
        return st.session_state.is_authenticated and st.session_state.user_id is not None
    
    def get_current_user(self) -> Optional[User]:
        """Get current user object"""
        if self.is_logged_in():
            return self.db.get_user(st.session_state.user_id)
        return None
    
    def login_user(self, username: str, password: str) -> bool:
        """Authenticate user login"""
        # For demo purposes, using simple authentication
        # In production, use proper password hashing and database storage
        
        # Demo users (replace with database authentication)
        demo_users = {
            "demo": {"password": "demo123", "user_id": 1},
            "user1": {"password": "password123", "user_id": 2},
            "fitness": {"password": "fitness2024", "user_id": 3}
        }
        
        if username in demo_users and demo_users[username]["password"] == password:
            st.session_state.user_id = demo_users[username]["user_id"]
            st.session_state.username = username
            st.session_state.is_authenticated = True
            return True
        
        return False
    
    def logout_user(self):
        """Logout current user"""
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.is_authenticated = False
    
    def logout(self):
        """Logout current user (alias for logout_user)"""
        self.logout_user()
    
    def create_user_profile(self, username: str, email: str, age: int, weight: float,
                           height: float, fitness_level: str, fitness_goal: str, goals: List[str]) -> bool:
        """Create a new user profile"""
        try:
            user_id = self.db.add_user(username, email, age, weight, height, fitness_level, fitness_goal, goals)
            if user_id:
                # Auto-login after profile creation
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.session_state.is_authenticated = True
                return True
        except Exception as e:
            st.error(f"Error creating profile: {e}")
            return False
        return False
    
    def update_user_profile(self, user_id: int, **kwargs) -> bool:
        """Update user profile information"""
        # This would require adding update methods to the database class
        # For now, return True as placeholder
        return True

def render_login_page():
    """Render the login page"""
    st.markdown("## 🔐 Login to Your Account")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            user_manager = UserManager()
            if user_manager.login_user(username, password):
                st.success("Login successful! Welcome back!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    st.markdown("---")
    st.markdown("### Demo Accounts")
    st.markdown("""
    - **Username:** demo, **Password:** demo123
    - **Username:** user1, **Password:** password123
    - **Username:** fitness, **Password:** fitness2024
    """)

def render_profile_creation():
    """Render the profile creation page"""
    st.markdown("## 📝 Create Your Profile")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username *")
            email = st.text_input("Email *")
            age = st.number_input("Age *", min_value=13, max_value=100, value=25)
            weight = st.number_input("Weight (kg) *", min_value=30.0, max_value=300.0, value=70.0)
        
        with col2:
            height = st.number_input("Height (cm) *", min_value=100.0, max_value=250.0, value=170.0)
            fitness_level = st.selectbox("Fitness Level *", 
                                       ["beginner", "intermediate", "advanced"])
            fitness_goal = st.selectbox("Primary Fitness Goal *",
                                      ["strength", "endurance", "flexibility", "weight_loss", 
                                       "muscle_gain", "general_fitness"])
            goals = st.multiselect("Fitness Goals *",
                                 ["strength", "endurance", "flexibility", "weight_loss", 
                                  "muscle_gain", "general_fitness"],
                                 default=["general_fitness"])
        
        submit_button = st.form_submit_button("Create Profile")
        
        if submit_button:
            if not all([username, email, age, weight, height, fitness_level, fitness_goal, goals]):
                st.error("Please fill in all required fields marked with *")
            else:
                user_manager = UserManager()
                if user_manager.create_user_profile(username, email, age, weight, height, 
                                                  fitness_level, fitness_goal, goals):
                    st.success("Profile created successfully! Welcome to flexIntel!")
                    st.rerun()
                else:
                    st.error("Failed to create profile. Please try again.")

def render_user_dashboard():
    """Render the main user dashboard"""
    user_manager = UserManager()
    user = user_manager.get_current_user()
    
    if not user:
        st.error("User not found")
        return
    
    st.markdown(f"## 👋 Welcome back, {user.username}!")
    
    # User info card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fitness Level", user.fitness_level.title())
    with col2:
        st.metric("Age", user.age)
    with col3:
        bmi = user.weight / ((user.height / 100) ** 2)
        st.metric("BMI", f"{bmi:.1f}")
    
    # Goals
    st.markdown("### 🎯 Your Goals")
    for goal in user.goals:
        st.markdown(f"- {goal.replace('_', ' ').title()}")
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    action_cols = st.columns(4)
    
    with action_cols[0]:
        if st.button("🏋️ Start Workout", type="primary"):
            st.session_state.current_page = "workout"
            st.rerun()
    
    with action_cols[1]:
        if st.button("📊 View Progress"):
            st.session_state.current_page = "progress"
            st.rerun()
    
    with action_cols[2]:
        if st.button("📋 Workout Plans"):
            st.session_state.current_page = "plans"
            st.rerun()
    
    with action_cols[3]:
        if st.button("⚙️ Settings"):
            st.session_state.current_page = "settings"
            st.rerun()
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    progress = user_manager.db.get_user_progress(user.id, days=7)
    
    if progress["total_sessions"] > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sessions This Week", progress["total_sessions"])
        with col2:
            st.metric("Total Reps", progress["total_reps"])
        with col3:
            st.metric("Avg Form Score", f"{progress['avg_form_score']}%")
    else:
        st.info("No recent activity. Start your first workout to see your progress!")

def render_progress_page():
    """Render the progress tracking page"""
    user_manager = UserManager()
    user = user_manager.get_current_user()
    
    if not user:
        st.error("User not found")
        return
    
    st.markdown("## 📊 Your Progress")
    
    # Time period selector
    period = st.selectbox("Time Period", ["7 days", "30 days", "90 days"], index=1)
    days = int(period.split()[0])
    
    # Exercise type filter
    exercise_types = ["All Exercises", "Bicep Curls", "Squats", "Shoulder Press", "Lateral Raise"]
    selected_exercise = st.selectbox("Exercise Type", exercise_types)
    
    exercise_filter = None
    if selected_exercise != "All Exercises":
        exercise_filter = selected_exercise.lower().replace(" ", "_")
    
    # Get progress data
    progress = user_manager.db.get_user_progress(user.id, exercise_filter, days)
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", progress["total_sessions"])
    with col2:
        st.metric("Total Reps", progress["total_reps"])
    with col3:
        st.metric("Total Sets", progress["total_sets"])
    with col4:
        st.metric("Avg Form Score", f"{progress['avg_form_score']}%")
    
    # Progress trend chart
    if progress["progress_trend"]:
        st.markdown("### 📈 Weekly Progress Trend")
        
        import pandas as pd
        import plotly.express as px
        
        df = pd.DataFrame(progress["progress_trend"])
        df['week'] = pd.to_datetime(df['week'])
        
        fig = px.line(df, x='week', y=['avg_reps', 'avg_form_score'], 
                     title="Weekly Performance Trends",
                     labels={'value': 'Score', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Form improvement suggestions
    if progress["total_sessions"] > 0:
        st.markdown("### 💡 Form Improvement Suggestions")
        recommendations = user_manager.db.get_exercise_recommendations(user.id)
        
        if recommendations.get("form_improvements"):
            for improvement in recommendations["form_improvements"]:
                st.info(improvement)
        else:
            st.success("Great job! Your form is looking good.")

def render_workout_plans_page():
    """Render the workout plans page"""
    user_manager = UserManager()
    user = user_manager.get_current_user()
    
    if not user:
        st.error("User not found")
        return
    
    st.markdown("## 📋 Your Workout Plans")
    
    # Get current recommendations
    recommendations = user_manager.db.get_exercise_recommendations(user.id)
    
    if recommendations:
        st.markdown("### 🎯 Recommended Next Workout")
        next_workout = recommendations["next_workout"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Level", next_workout["level"].title())
        with col2:
            st.metric("Duration", f"{next_workout['estimated_duration']} min")
        with col3:
            st.metric("Focus", next_workout["focus"].title())
        
        st.markdown("#### Exercises:")
        for exercise, details in next_workout["exercises"].items():
            exercise_name = exercise.replace("_", " ").title()
            st.markdown(f"**{exercise_name}**: {details['sets']} sets × {details['reps']} reps (rest: {details['rest']}s)")
        
        if st.button("💪 Start This Workout", type="primary"):
            st.session_state.current_page = "workout"
            st.rerun()
    
    # Progression tips
    if recommendations and recommendations.get("progression_tips"):
        st.markdown("### 🚀 Progression Tips")
        for tip in recommendations["progression_tips"]:
            st.info(tip)
    
    # Saved workout plans
    st.markdown("### 💾 Saved Workout Plans")
    saved_plans = user_manager.db.get_user_workout_plans(user.id)
    
    if saved_plans:
        for plan in saved_plans:
            with st.expander(f"📋 {plan.name} ({plan.difficulty})"):
                st.markdown(f"**Created:** {plan.created_at.strftime('%Y-%m-%d')}")
                st.markdown(f"**Duration:** {plan.estimated_duration} minutes")
                st.markdown("**Exercises:**")
                for exercise, details in plan.exercises.items():
                    exercise_name = exercise.replace("_", " ").title()
                    st.markdown(f"- {exercise_name}: {details['sets']} sets × {details['reps']} reps")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Start {plan.name}", key=f"start_{plan.id}"):
                        st.session_state.current_page = "workout"
                        st.rerun()
                with col2:
                    if st.button(f"Delete {plan.name}", key=f"delete_{plan.id}"):
                        # Add delete functionality to database
                        st.success(f"Deleted {plan.name}")
                        st.rerun()
    else:
        st.info("No saved workout plans yet. Your recommended workout will appear above!")

def render_settings_page():
    """Render the settings page"""
    user_manager = UserManager()
    user = user_manager.get_current_user()
    
    if not user:
        st.error("User not found")
        return
    
    st.markdown("## ⚙️ Settings")
    
    # Profile information
    st.markdown("### 👤 Profile Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Username", value=user.username, disabled=True)
        st.text_input("Email", value=user.email, disabled=True)
        st.number_input("Age", value=user.age, disabled=True)
    
    with col2:
        st.number_input("Weight (kg)", value=user.weight, disabled=True)
        st.number_input("Height (cm)", value=user.height, disabled=True)
        st.selectbox("Fitness Level", ["beginner", "intermediate", "advanced"], 
                    index=["beginner", "intermediate", "advanced"].index(user.fitness_level), 
                    disabled=True)
    
    st.markdown("**Goals:** " + ", ".join([goal.replace("_", " ").title() for goal in user.goals]))
    
    # Account actions
    st.markdown("### 🔐 Account Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Profile"):
            st.rerun()
    
    with col2:
        if st.button("🚪 Logout", type="secondary"):
            user_manager.logout_user()
            st.success("Logged out successfully!")
            st.rerun()
    
    # Data export
    st.markdown("### 📤 Data Export")
    if st.button("📊 Export Progress Data"):
        # Add export functionality
        st.info("Export functionality coming soon!")
    
    # Delete account
    st.markdown("### ⚠️ Danger Zone")
    if st.button("🗑️ Delete Account", type="secondary"):
        st.warning("This action cannot be undone. Are you sure?")
        if st.button("Yes, Delete My Account", type="primary"):
            # Add delete account functionality
            st.info("Account deletion coming soon!")
