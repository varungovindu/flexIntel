#!/usr/bin/env python3
"""
Demo data population script for flexIntel
This script creates sample users and exercise data for testing the application.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from database import FitnessDatabase
from datetime import datetime, timedelta
import random

def create_demo_data():
    """Create demo data for testing"""
    db = FitnessDatabase()
    
    print("🏋️ Creating demo data for flexIntel...")
    
    # Create demo users
    demo_users = [
        {
            "username": "demo",
            "email": "demo@flexintel.com",
            "age": 25,
            "weight": 70.0,
            "height": 175.0,
            "fitness_level": "beginner",
            "fitness_goal": "general_fitness",
            "goals": ["general_fitness", "strength"]
        },
        {
            "username": "user1",
            "email": "user1@flexintel.com",
            "age": 30,
            "weight": 75.0,
            "height": 180.0,
            "fitness_level": "intermediate",
            "fitness_goal": "strength",
            "goals": ["strength", "muscle_gain"]
        },
        {
            "username": "fitness",
            "email": "fitness@flexintel.com",
            "age": 28,
            "weight": 65.0,
            "height": 165.0,
            "fitness_level": "advanced",
            "fitness_goal": "endurance",
            "goals": ["endurance", "flexibility", "strength"]
        }
    ]
    
    user_ids = {}
    for user_data in demo_users:
        try:
            user_id = db.add_user(**user_data)
            user_ids[user_data["username"]] = user_id
            print(f"✅ Created user: {user_data['username']} (ID: {user_id})")
        except Exception as e:
            print(f"❌ Failed to create user {user_data['username']}: {e}")
    
    # Create sample exercise sessions
    exercise_types = ["Bicep Curls", "Squats", "Shoulder Press", "Lateral Raise", "Pushups", "Plank"]
    
    for username, user_id in user_ids.items():
        print(f"\n📊 Creating exercise data for {username}...")
        
        # Create sessions over the last 30 days
        for days_ago in range(30, 0, -2):  # Every 2 days
            if random.random() < 0.7:  # 70% chance of working out
                exercise_type = random.choice(exercise_types)
                start_time = datetime.now() - timedelta(days=days_ago, hours=random.randint(8, 20))
                duration = random.randint(15, 45)  # 15-45 minutes
                end_time = start_time + timedelta(minutes=duration)
                
                # Generate realistic exercise data
                total_reps = random.randint(20, 80)
                total_sets = max(1, total_reps // 10)
                
                # Form score based on fitness level
                base_form = {"beginner": 70, "intermediate": 80, "advanced": 90}
                user_level = next(u["fitness_level"] for u in demo_users if u["username"] == username)
                form_score = base_form[user_level] + random.randint(-10, 10)
                form_score = max(60, min(100, form_score))  # Clamp between 60-100
                
                # Estimate calories
                calories_burned = duration * 4.0 * (form_score / 100) * (1.0 + total_reps / 100 * 0.3)
                
                try:
                    session_id = db.add_exercise_session(
                        user_id=user_id,
                        exercise_type=exercise_type,
                        start_time=start_time,
                        end_time=end_time,
                        total_reps=total_reps,
                        total_sets=total_sets,
                        form_score=form_score,
                        calories_burned=round(calories_burned, 1),
                        duration_minutes=duration
                    )
                    
                    # Create individual rep data
                    for rep_num in range(1, total_reps + 1):
                        set_num = (rep_num - 1) // 10 + 1
                        
                        # Vary form score slightly for each rep
                        rep_form_score = max(60, min(100, form_score + random.randint(-5, 5)))
                        rep_time = random.uniform(1.5, 3.0)
                        
                        # Generate realistic joint angles
                        angles = {
                            "elbow_angle": random.randint(80, 170),
                            "shoulder_angle": random.randint(70, 160),
                            "knee_angle": random.randint(60, 180)
                        }
                        
                        # Generate feedback based on form score
                        feedback = []
                        if rep_form_score < 75:
                            feedback.append("Focus on form")
                        if rep_form_score < 80:
                            feedback.append("Control the movement")
                        if rep_form_score >= 90:
                            feedback.append("Excellent form!")
                        
                        db.add_exercise_rep(
                            session_id=session_id,
                            rep_number=rep_num,
                            set_number=set_num,
                            form_score=rep_form_score,
                            rep_time=rep_time,
                            angles=angles,
                            feedback=feedback
                        )
                    
                    print(f"  ✅ Created {exercise_type} session: {total_reps} reps, {form_score}% form")
                    
                except Exception as e:
                    print(f"  ❌ Failed to create session: {e}")
    
    # Create sample workout plans
    print(f"\n📋 Creating workout plans...")
    
    workout_templates = {
        "beginner": {
            "name": "Beginner Strength Builder",
            "exercises": {
                "bicep_curls": {"sets": 2, "reps": 8, "rest": 60},
                "squats": {"sets": 2, "reps": 10, "rest": 90},
                "shoulder_press": {"sets": 2, "reps": 8, "rest": 60},
                "pushups": {"sets": 2, "reps": 5, "rest": 60},
                "plank": {"sets": 2, "reps": 15, "rest": 60}
            },
            "difficulty": "beginner",
            "estimated_duration": 30
        },
        "intermediate": {
            "name": "Intermediate Power Program",
            "exercises": {
                "bicep_curls": {"sets": 3, "reps": 12, "rest": 45},
                "squats": {"sets": 3, "reps": 15, "rest": 75},
                "shoulder_press": {"sets": 3, "reps": 10, "rest": 45},
                "lateral_raise": {"sets": 3, "reps": 12, "rest": 45},
                "pushups": {"sets": 3, "reps": 10, "rest": 45},
                "plank": {"sets": 3, "reps": 30, "rest": 45},
                "mountain_climber": {"sets": 3, "reps": 20, "rest": 45}
            },
            "difficulty": "intermediate",
            "estimated_duration": 45
        },
        "advanced": {
            "name": "Advanced Performance",
            "exercises": {
                "bicep_curls": {"sets": 4, "reps": 15, "rest": 30},
                "squats": {"sets": 4, "reps": 20, "rest": 60},
                "shoulder_press": {"sets": 4, "reps": 12, "rest": 30},
                "lateral_raise": {"sets": 4, "reps": 15, "rest": 30},
                "pushups": {"sets": 4, "reps": 15, "rest": 30},
                "plank": {"sets": 4, "reps": 45, "rest": 30},
                "mountain_climber": {"sets": 4, "reps": 30, "rest": 30}
            },
            "difficulty": "advanced",
            "estimated_duration": 60
        }
    }
    
    for username, user_id in user_ids.items():
        user_level = next(u["fitness_level"] for u in demo_users if u["username"] == username)
        template = workout_templates[user_level]
        
        try:
            plan_id = db.save_workout_plan(
                user_id=user_id,
                name=template["name"],
                exercises=template["exercises"],
                difficulty=template["difficulty"],
                estimated_duration=template["estimated_duration"]
            )
            print(f"  ✅ Created workout plan for {username}: {template['name']}")
        except Exception as e:
            print(f"  ❌ Failed to create workout plan for {username}: {e}")
    
    print(f"\n🎉 Demo data creation completed!")
    print(f"📊 Created {len(user_ids)} users with exercise data")
    print(f"🏋️ Exercise sessions span the last 30 days")
    print(f"📋 Custom workout plans for each fitness level")
    print(f"\n🚀 You can now login with any demo account to test the app!")

if __name__ == "__main__":
    create_demo_data()
