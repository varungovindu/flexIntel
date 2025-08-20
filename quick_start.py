#!/usr/bin/env python3
"""
Quick start script for flexIntel
This script helps users set up and run the application quickly.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "streamlit", "pandas", "plotly", "numpy", "opencv-python", "mediapipe"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_demo_data():
    """Create demo data for testing"""
    print("\n🏋️ Creating demo data...")
    try:
        subprocess.check_call([sys.executable, "demo_data.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create demo data: {e}")
        return False

def run_application():
    """Run the Streamlit application"""
    print("\n🚀 Starting flexIntel...")
    print("📱 The app will open in your default web browser")
    print("🔑 Use demo accounts to test:")
    print("   - demo / demo123")
    print("   - user1 / password123")
    print("   - fitness / fitness2024")
    print("\n⏹️ Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped. Goodbye!")

def main():
    """Main quick start function"""
    print("🏋️‍♂️ flexIntel - Quick Start")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        install_choice = input("Install missing dependencies? (y/n): ").lower()
        
        if install_choice == 'y':
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually:")
                print("pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("❌ Please install missing dependencies before continuing")
            sys.exit(1)
    
    # Check if demo data exists
    db_file = Path("fitness_data.db")
    if not db_file.exists():
        print("\n📊 No database found. Creating demo data...")
        if not create_demo_data():
            print("❌ Failed to create demo data")
            sys.exit(1)
    else:
        print("✅ Database found")
    
    # Run the application
    run_application()

if __name__ == "__main__":
    main()
