# 🏋️‍♂️ flexIntel - AI-Powered Fitness Coach

A comprehensive AI-powered fitness application that tracks your exercises in real-time using computer vision and provides personalized workout recommendations.

## ✨ Features

- **Real-time Exercise Tracking**: AI-powered pose detection for accurate form analysis
- **Multiple Exercise Types**: Support for 7 different exercises with consistent interfaces
- **Professional UI**: Modern, responsive Streamlit interface with beautiful styling
- **Progress Tracking**: Comprehensive workout history and performance analytics
- **Personalized Workouts**: Adaptive workout plans based on fitness level and goals
- **Form Feedback**: Real-time guidance to improve exercise technique

## 🎯 Supported Exercises

### 1. 💪 Bicep Curls
- **Description**: Tracks bicep curl repetitions with form analysis
- **Key Features**: Angle detection, rep counting, form scoring
- **Parameters**: Top angle, bottom angle, smoothing factor

### 2. 🏋️ Squats
- **Description**: Monitors squat depth and form
- **Key Features**: Knee angle tracking, depth validation, rep counting
- **Parameters**: Squat depth angles, smoothing factor

### 3. 🏋️‍♂️ Shoulder Press
- **Description**: Tracks overhead press movements
- **Key Features**: Arm position detection, form validation
- **Parameters**: Offset ratios, smoothing factor

### 4. 🦾 Lateral Raise
- **Description**: Monitors lateral shoulder raises
- **Key Features**: Arm angle tracking, movement validation
- **Parameters**: Top/bottom angles, smoothing factor

### 5. 🦵 Pushups
- **Description**: Tracks pushup form and repetitions
- **Key Features**: Body alignment, elbow angle monitoring, rep counting
- **Parameters**: Top/bottom angles, smoothing factor

### 6. 🧘 Plank
- **Description**: Monitors plank hold duration and form
- **Key Features**: Body alignment tracking, stability scoring, timer
- **Parameters**: Hold time, drift thresholds, smoothing factor

### 7. 🏃 Mountain Climber
- **Description**: Tracks mountain climber movements
- **Key Features**: Knee movement detection, plank position validation
- **Parameters**: Knee angle ranges, smoothing factor

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or video input device
- Windows 10/11 (tested on Windows)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flexIntel
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📱 Usage

### Starting the App
1. Run `streamlit run app.py`
2. Open your browser to the displayed URL
3. Create an account or log in with demo credentials

### Using Exercises
1. Navigate to the **Workout** tab
2. Select your desired exercise from the dropdown
3. Configure video settings (camera, FPS, model complexity)
4. Adjust exercise-specific parameters if needed
5. Click **Start Exercise** to begin tracking
6. Follow the on-screen feedback for optimal form

### Demo Accounts
Use these pre-created accounts for testing:
- **Username**: `demo` | **Level**: Beginner
- **Username**: `user1` | **Level**: Intermediate  
- **Username**: `fitness` | **Level**: Advanced

## ⚙️ Configuration

### Video Settings
- **Input Source**: Webcam (index 0) or video file path
- **FPS**: Frame rate for capture and processing
- **Model Complexity**: MediaPipe model accuracy (0-2)
- **Detection Confidence**: Minimum confidence for pose detection

### Exercise Parameters
Each exercise has specific parameters that can be adjusted:
- **Angles**: Threshold angles for movement detection
- **Smoothing**: EMA smoothing factor for stable tracking
- **Hold Frames**: Frames required to register position changes
- **Thresholds**: Form validation limits

## 🏗️ Architecture

### Core Components
- **Streamlit Frontend**: Modern web interface with real-time updates
- **MediaPipe Integration**: Advanced pose detection and tracking
- **Exercise Modules**: Specialized tracking for each exercise type
- **Database Layer**: SQLite storage for user data and progress
- **AI Analytics**: Form scoring and performance analysis

### File Structure
```
flexIntel/
├── app.py                 # Main Streamlit application
├── database.py            # Database operations and models
├── exercises/             # Exercise tracking modules
│   ├── bicep_counter.py
│   ├── squat.py
│   ├── shoulder_press.py
│   ├── lateral_raise.py
│   ├── pushup_tracker.py
│   ├── plank_tracker.py
│   └── mountain_climber_tracker.py
├── templates/             # HTML templates
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Customization

### Adding New Exercises
1. Create a new Python file in the `exercises/` directory
2. Follow the established interface pattern:
   - Configuration dataclasses
   - Pose detection integration
   - State machine for tracking
   - Professional overlay drawing
   - Command-line argument parsing
3. Update `app.py` to include the new exercise
4. Add exercise parameters and command building logic

### Modifying Existing Exercises
- Adjust angle thresholds in the configuration
- Modify form validation logic
- Customize visual feedback and overlays
- Update parameter ranges in the Streamlit interface

## 📊 Performance Tips

### Optimal Settings
- **Model Complexity**: Use 1 for most cases, 2 for high accuracy
- **Detection Confidence**: Start with 0.5, increase for stability
- **FPS**: 30 FPS provides good balance of accuracy and performance
- **Lighting**: Ensure good lighting for better pose detection

### Troubleshooting
- **Poor Detection**: Increase detection confidence, improve lighting
- **Lag**: Reduce FPS, use lower model complexity
- **Inaccurate Tracking**: Adjust exercise parameters, check camera positioning

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Advanced pose detection and tracking
- **Streamlit**: Beautiful web application framework
- **OpenCV**: Computer vision capabilities
- **OpenAI**: AI assistance for development

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Made with ❤️ for fitness enthusiasts everywhere**

*Transform your workouts with AI-powered precision tracking!*
