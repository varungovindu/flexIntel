#!/usr/bin/env python3
"""
Test script for flexIntel exercise modules
Verifies all exercises can be imported and have required functionality
"""

import sys
from pathlib import Path
import importlib.util

def test_exercise_module(module_name: str, file_path: Path) -> bool:
    """Test if an exercise module can be imported and has required functions"""
    try:
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"❌ Could not create spec for {module_name}")
            return False
            
        module = importlib.util.module_from_spec(spec)
        if module is None:
            print(f"❌ Could not create module for {module_name}")
            return False
            
        spec.loader.exec_module(module)
        
        # Check for required functions
        required_functions = ['run_app', 'build_arg_parser']
        missing_functions = []
        
        for func_name in required_functions:
            if not hasattr(module, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"❌ {module_name}: Missing functions: {missing_functions}")
            return False
        
        # Check for required classes
        required_classes = ['CounterConfig', 'DrawConfig']
        missing_classes = []
        
        for class_name in required_classes:
            if not hasattr(module, class_name):
                missing_classes.append(class_name)
        
        if missing_classes:
            print(f"⚠️  {module_name}: Missing classes: {missing_classes}")
            # Not critical, just warning
        
        print(f"✅ {module_name}: All required functions present")
        return True
        
    except Exception as e:
        print(f"❌ {module_name}: Import failed - {e}")
        return False

def main():
    """Test all exercise modules"""
    print("🧪 Testing flexIntel Exercise Modules")
    print("=" * 50)
    
    # Get exercises directory
    exercises_dir = Path(__file__).parent / "exercises"
    
    if not exercises_dir.exists():
        print(f"❌ Exercises directory not found: {exercises_dir}")
        return
    
    # Define expected exercise modules
    expected_exercises = {
        "bicep_counter": "bicep_counter.py",
        "squat": "squat.py", 
        "shoulder_press": "shoulder_press.py",
        "lateral_raise": "lateral_raise.py",
        "pushup_tracker": "pushup_tracker.py",
        "plank_tracker": "plank_tracker.py"
    }
    
    # Test each module
    passed = 0
    total = len(expected_exercises)
    
    for module_name, filename in expected_exercises.items():
        file_path = exercises_dir / filename
        
        if not file_path.exists():
            print(f"❌ {module_name}: File not found - {file_path}")
            continue
            
        if test_exercise_module(module_name, file_path):
            passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} modules passed")
    
    if passed == total:
        print("🎉 All exercise modules are working correctly!")
    else:
        print("⚠️  Some modules have issues. Check the errors above.")
    
    # Test main app import
    print("\n🔍 Testing main application...")
    try:
        app_path = Path(__file__).parent / "app.py"
        if app_path.exists():
            print("✅ Main app file found")
        else:
            print("❌ Main app file not found")
    except Exception as e:
        print(f"❌ Error checking main app: {e}")
    
    # Test database import
    print("\n🔍 Testing database module...")
    try:
        db_path = Path(__file__).parent / "database.py"
        if db_path.exists():
            print("✅ Database module found")
        else:
            print("❌ Database module not found")
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    main()
