import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class User:
    id: int
    username: str
    email: str
    age: int
    weight: float
    height: float
    fitness_level: str  # beginner, intermediate, advanced
    fitness_goal: str  # strength, endurance, flexibility, weight_loss, etc.
    goals: List[str]  # strength, endurance, flexibility, etc.
    created_at: datetime

@dataclass
class ExerciseSession:
    id: int
    user_id: int
    exercise_type: str
    start_time: datetime
    end_time: datetime
    total_reps: int
    total_sets: int
    form_score: float  # 0-100
    calories_burned: float
    duration_minutes: float

@dataclass
class ExerciseRep:
    id: int
    session_id: int
    rep_number: int
    set_number: int
    form_score: float
    rep_time: float  # seconds
    angles: Dict[str, float]  # joint angles during rep
    feedback: List[str]  # form feedback

@dataclass
class WorkoutPlan:
    id: int
    user_id: int
    name: str
    exercises: List[Dict]
    difficulty: str
    estimated_duration: int  # minutes
    created_at: datetime

class FitnessDatabase:
    def __init__(self, db_path: str = "fitness_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    age INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    height REAL NOT NULL,
                    fitness_level TEXT NOT NULL,
                    fitness_goal TEXT NOT NULL,
                    goals TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exercise_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    exercise_type TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    total_reps INTEGER NOT NULL,
                    total_sets INTEGER NOT NULL,
                    form_score REAL NOT NULL,
                    calories_burned REAL NOT NULL,
                    duration_minutes REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exercise_reps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    rep_number INTEGER NOT NULL,
                    set_number INTEGER NOT NULL,
                    form_score REAL NOT NULL,
                    rep_time REAL NOT NULL,
                    angles TEXT NOT NULL,
                    feedback TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES exercise_sessions (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workout_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    exercises TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    estimated_duration INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # Calorie tracker logs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calorie_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    log_date DATE NOT NULL,
                    food TEXT NOT NULL,
                    calories REAL NOT NULL,
                    protein_g REAL DEFAULT 0,
                    carbs_g REAL DEFAULT 0,
                    fats_g REAL DEFAULT 0,
                    notes TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.commit()
    
    def add_user(self, username: str, email: str, age: int, weight: float, 
                 height: float, fitness_level: str, fitness_goal: str, goals: List[str]) -> int:
        """Add a new user and return user ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO users (username, email, age, weight, height, fitness_level, fitness_goal, goals)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (username, email, age, weight, height, fitness_level, fitness_goal, json.dumps(goals)))
            return cursor.lastrowid
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return User(
                    id=row[0], username=row[1], email=row[2], age=row[3],
                    weight=row[4], height=row[5], fitness_level=row[6],
                    fitness_goal=row[7], goals=json.loads(row[8]), created_at=datetime.fromisoformat(row[9])
                )
            return None
    
    def add_exercise_session(self, user_id: int, exercise_type: str, 
                           start_time: datetime, end_time: datetime,
                           total_reps: int, total_sets: int, form_score: float,
                           calories_burned: float, duration_minutes: float) -> int:
        """Add exercise session and return session ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO exercise_sessions 
                (user_id, exercise_type, start_time, end_time, total_reps, 
                 total_sets, form_score, calories_burned, duration_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, exercise_type, start_time.isoformat(), end_time.isoformat(),
                  total_reps, total_sets, form_score, calories_burned, duration_minutes))
            return cursor.lastrowid
    
    def add_exercise_rep(self, session_id: int, rep_number: int, set_number: int,
                        form_score: float, rep_time: float, angles: Dict[str, float],
                        feedback: List[str]) -> int:
        """Add individual rep data and return rep ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO exercise_reps 
                (session_id, rep_number, set_number, form_score, rep_time, angles, feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, rep_number, set_number, form_score, rep_time,
                  json.dumps(angles), json.dumps(feedback)))
            return cursor.lastrowid
    
    def get_user_progress(self, user_id: int, exercise_type: str = None, 
                         days: int = 30) -> Dict:
        """Get user progress over specified days"""
        with sqlite3.connect(self.db_path) as conn:
            start_date = datetime.now() - timedelta(days=days)
            
            if exercise_type:
                cursor = conn.execute("""
                    SELECT * FROM exercise_sessions 
                    WHERE user_id = ? AND exercise_type = ? AND start_time >= ?
                    ORDER BY start_time DESC
                """, (user_id, exercise_type, start_date.isoformat()))
            else:
                cursor = conn.execute("""
                    SELECT * FROM exercise_sessions 
                    WHERE user_id = ? AND start_time >= ?
                    ORDER BY start_time DESC
                """, (user_id, start_date.isoformat()))
            
            sessions = cursor.fetchall()
            
            if not sessions:
                return {
                    "total_sessions": 0,
                    "total_reps": 0,
                    "total_sets": 0,
                    "avg_form_score": 0,
                    "total_calories": 0,
                    "total_duration": 0,
                    "progress_trend": []
                }
            
            # Calculate progress metrics
            total_sessions = len(sessions)
            total_reps = sum(session[5] for session in sessions)
            total_sets = sum(session[6] for session in sessions)
            avg_form_score = sum(session[7] for session in sessions) / total_sessions
            total_calories = sum(session[8] for session in sessions)
            total_duration = sum(session[9] for session in sessions)
            
            # Calculate progress trend (weekly averages)
            weekly_data = {}
            for session in sessions:
                week_start = datetime.fromisoformat(session[3]).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) - timedelta(days=datetime.fromisoformat(session[3]).weekday())
                week_key = week_start.isoformat()
                
                if week_key not in weekly_data:
                    weekly_data[week_key] = {"reps": [], "form_scores": []}
                
                weekly_data[week_key]["reps"].append(session[5])
                weekly_data[week_key]["form_scores"].append(session[7])
            
            progress_trend = []
            for week, data in sorted(weekly_data.items()):
                progress_trend.append({
                    "week": week,
                    "avg_reps": np.mean(data["reps"]),
                    "avg_form_score": np.mean(data["form_scores"])
                })
            
            return {
                "total_sessions": total_sessions,
                "total_reps": total_reps,
                "total_sets": total_sets,
                "avg_form_score": round(avg_form_score, 2),
                "total_calories": total_calories,
                "total_duration": total_duration,
                "progress_trend": progress_trend
            }
    
    def get_exercise_recommendations(self, user_id: int) -> Dict:
        """Generate personalized exercise recommendations based on user data"""
        user = self.get_user(user_id)
        if not user:
            return {}
        
        progress = self.get_user_progress(user_id)
        
        # Analyze user's current performance
        current_level = self._assess_user_level(progress, user.fitness_level)
        
        # Generate recommendations based on level and goals
        recommendations = {
            "current_level": current_level,
            "next_workout": self._generate_workout_plan(user, current_level, progress),
            "form_improvements": self._get_form_improvements(user_id),
            "progression_tips": self._get_progression_tips(current_level, user.goals)
        }
        
        return recommendations
    
    def _assess_user_level(self, progress: Dict, base_level: str) -> str:
        """Assess user's current fitness level based on progress"""
        if progress["total_sessions"] < 5:
            return "beginner"
        
        avg_form_score = progress["avg_form_score"]
        total_reps = progress["total_reps"]
        
        if base_level == "beginner":
            if avg_form_score > 85 and total_reps > 100:
                return "intermediate"
        elif base_level == "intermediate":
            if avg_form_score > 90 and total_reps > 300:
                return "advanced"
            elif avg_form_score < 70:
                return "beginner"
        elif base_level == "advanced":
            if avg_form_score < 80:
                return "intermediate"
        
        return base_level
    
    def _generate_workout_plan(self, user: User, level: str, progress: Dict) -> Dict:
        """Generate a personalized workout plan"""
        # Base workout templates
        workout_templates = {
            "beginner": {
                "bicep_curls": {"sets": 2, "reps": 8, "rest": 60},
                "squats": {"sets": 2, "reps": 10, "rest": 90},
                "shoulder_press": {"sets": 2, "reps": 8, "rest": 60},
                "lateral_raise": {"sets": 2, "reps": 10, "rest": 60}
            },
            "intermediate": {
                "bicep_curls": {"sets": 3, "reps": 12, "rest": 45},
                "squats": {"sets": 3, "reps": 15, "rest": 75},
                "shoulder_press": {"sets": 3, "reps": 10, "rest": 45},
                "lateral_raise": {"sets": 3, "reps": 12, "rest": 45}
            },
            "advanced": {
                "bicep_curls": {"sets": 4, "reps": 15, "rest": 30},
                "squats": {"sets": 4, "reps": 20, "rest": 60},
                "shoulder_press": {"sets": 4, "reps": 12, "rest": 30},
                "lateral_raise": {"sets": 4, "reps": 15, "rest": 30}
            }
        }
        
        # Adjust based on recent performance
        template = workout_templates[level].copy()
        
        if progress["avg_form_score"] > 90:
            # User is performing well, increase difficulty
            for exercise in template:
                template[exercise]["reps"] += 2
                template[exercise]["rest"] = max(15, template[exercise]["rest"] - 10)
        elif progress["avg_form_score"] < 75:
            # User needs to focus on form, reduce difficulty
            for exercise in template:
                template[exercise]["reps"] = max(5, template[exercise]["reps"] - 2)
                template[exercise]["rest"] += 15
        
        return {
            "level": level,
            "exercises": template,
            "estimated_duration": sum(ex["sets"] * 2 + ex["rest"] for ex in template.values()) // 60,
            "focus": "form" if progress["avg_form_score"] < 80 else "progression"
        }
    
    def _get_form_improvements(self, user_id: int) -> List[str]:
        """Get specific form improvement suggestions based on recent sessions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT er.feedback, er.form_score, es.exercise_type
                FROM exercise_reps er
                JOIN exercise_sessions es ON er.session_id = es.id
                WHERE es.user_id = ? AND er.form_score < 80
                ORDER BY es.start_time DESC
                LIMIT 10
            """, (user_id,))
            
            recent_feedback = cursor.fetchall()
            
            improvements = []
            if recent_feedback:
                # Analyze common feedback patterns
                feedback_counts = {}
                for feedback, score, exercise in recent_feedback:
                    feedback_list = json.loads(feedback)
                    for item in feedback_list:
                        if item not in feedback_counts:
                            feedback_counts[item] = 0
                        feedback_counts[item] += 1
                
                # Get top 3 most common issues
                top_issues = sorted(feedback_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                for issue, count in top_issues:
                    improvements.append(f"Focus on: {issue} (appeared {count} times)")
            
            return improvements
    
    def _get_progression_tips(self, level: str, goals: List[str]) -> List[str]:
        """Get progression tips based on user level and goals"""
        tips = []
        
        if level == "beginner":
            tips.extend([
                "Focus on proper form before increasing intensity",
                "Aim for 2-3 workouts per week",
                "Start with bodyweight exercises to build foundation"
            ])
        elif level == "intermediate":
            tips.extend([
                "Gradually increase weight or resistance",
                "Add variety to prevent plateaus",
                "Consider supersets to increase workout intensity"
            ])
        elif level == "advanced":
            tips.extend([
                "Incorporate advanced techniques like drop sets",
                "Focus on periodization and recovery",
                "Consider working with a trainer for specialized programs"
            ])
        
        if "strength" in goals:
            tips.append("Focus on progressive overload - gradually increase weight")
        if "endurance" in goals:
            tips.append("Increase rep ranges and reduce rest periods")
        if "flexibility" in goals:
            tips.append("Include stretching and mobility work in your routine")
        
        return tips
    
    def save_workout_plan(self, user_id: int, name: str, exercises: List[Dict],
                         difficulty: str, estimated_duration: int) -> int:
        """Save a workout plan for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO workout_plans (user_id, name, exercises, difficulty, estimated_duration)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, name, json.dumps(exercises), difficulty, estimated_duration))
            return cursor.lastrowid
    
    def get_user_workout_plans(self, user_id: int) -> List[WorkoutPlan]:
        """Get all workout plans for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM workout_plans WHERE user_id = ? ORDER BY created_at DESC
            """, (user_id,))
            
            plans = []
            for row in cursor.fetchall():
                plans.append(WorkoutPlan(
                    id=row[0], user_id=row[1], name=row[2],
                    exercises=json.loads(row[3]), difficulty=row[4],
                    estimated_duration=row[5], created_at=datetime.fromisoformat(row[6])
                ))
            
            return plans

    # ---- Calorie tracker methods ----
    def add_calorie_log(self, user_id: int, log_date: datetime, food: str, calories: float,
                        protein_g: float = 0.0, carbs_g: float = 0.0, fats_g: float = 0.0,
                        notes: str = "") -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO calorie_logs (user_id, log_date, food, calories, protein_g, carbs_g, fats_g, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, log_date.date().isoformat(), food, calories, protein_g, carbs_g, fats_g, notes),
            )
            return cursor.lastrowid

    def get_calorie_summary(self, user_id: int, days: int = 7) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
            cursor = conn.execute(
                """
                SELECT log_date, SUM(calories), SUM(protein_g), SUM(carbs_g), SUM(fats_g)
                FROM calorie_logs
                WHERE user_id = ? AND log_date >= ?
                GROUP BY log_date
                ORDER BY log_date DESC
                """,
                (user_id, start_date),
            )
            rows = cursor.fetchall()
            history = [
                {
                    "date": r[0],
                    "calories": float(r[1] or 0),
                    "protein_g": float(r[2] or 0),
                    "carbs_g": float(r[3] or 0),
                    "fats_g": float(r[4] or 0),
                }
                for r in rows
            ]
            total_kcal = sum(item[1] for item in rows) if rows else 0.0
            return {
                "days": days,
                "total_kcal": float(total_kcal),
                "history": history,
            }
