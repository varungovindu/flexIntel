import React, { useState, useMemo, useCallback, memo } from 'react';
import { FixedSizeList as List } from 'react-window';
import { usePoseDetection } from '../hooks/usePoseDetection';
import { Exercise, PerformanceMetrics } from '../types';
import PoseDetection from './PoseDetection';
import ExerciseList from './ExerciseList';
import PerformanceDashboard from './PerformanceDashboard';
import WorkoutSession from './WorkoutSession';

// Memoized exercise data to prevent unnecessary re-renders
const EXERCISES: Exercise[] = [
  {
    id: '1',
    name: 'Push-ups',
    description: 'Classic bodyweight exercise for upper body strength',
    difficulty: 'beginner',
    category: 'strength',
    instructions: [
      'Start in a plank position',
      'Lower your body until chest nearly touches the floor',
      'Push back up to starting position'
    ],
    targetMuscles: ['chest', 'triceps', 'shoulders']
  },
  {
    id: '2',
    name: 'Squats',
    description: 'Fundamental lower body exercise',
    difficulty: 'beginner',
    category: 'strength',
    instructions: [
      'Stand with feet shoulder-width apart',
      'Lower your body as if sitting back into a chair',
      'Keep your chest up and knees behind toes'
    ],
    targetMuscles: ['quadriceps', 'glutes', 'hamstrings']
  },
  {
    id: '3',
    name: 'Plank',
    description: 'Core stability exercise',
    difficulty: 'beginner',
    category: 'strength',
    instructions: [
      'Hold a push-up position with arms straight',
      'Keep your body in a straight line',
      'Engage your core muscles'
    ],
    targetMuscles: ['core', 'shoulders', 'back']
  }
];

const FitnessCoach: React.FC = memo(() => {
  const [currentExercise, setCurrentExercise] = useState<Exercise | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    accuracy: 0,
    reps: 0,
    form: 'good',
    feedback: []
  });
  const [isWorkoutActive, setIsWorkoutActive] = useState(false);

  // Memoized pose detection hook
  const poseDetection = usePoseDetection({
    onAnalysis: useCallback((analysis: any) => {
      setPerformanceMetrics(prev => ({
        ...prev,
        accuracy: analysis.posture.score,
        form: analysis.posture.score > 80 ? 'good' : 
              analysis.posture.score > 60 ? 'needs_improvement' : 'poor',
        feedback: analysis.posture.feedback
      }));
    }, [])
  });

  // Memoized filtered exercises
  const filteredExercises = useMemo(() => {
    return EXERCISES.filter(exercise => 
      exercise.difficulty === 'beginner' || exercise.difficulty === 'intermediate'
    );
  }, []);

  const handleExerciseSelect = useCallback((exercise: Exercise) => {
    setCurrentExercise(exercise);
    setIsWorkoutActive(true);
  }, []);

  const handleWorkoutEnd = useCallback(() => {
    setIsWorkoutActive(false);
    setCurrentExercise(null);
    setPerformanceMetrics({
      accuracy: 0,
      reps: 0,
      form: 'good',
      feedback: []
    });
  }, []);

  if (isWorkoutActive && currentExercise) {
    return (
      <div className="fitness-coach">
        <WorkoutSession
          exercise={currentExercise}
          poseDetection={poseDetection}
          performanceMetrics={performanceMetrics}
          onEnd={handleWorkoutEnd}
        />
      </div>
    );
  }

  return (
    <div className="fitness-coach">
      <div className="coach-header">
        <h2>Welcome to flexIntel</h2>
        <p>Select an exercise to begin your AI-powered workout</p>
      </div>
      
      <div className="coach-content">
        <div className="pose-detection-section">
          <PoseDetection poseDetection={poseDetection} />
        </div>
        
        <div className="exercise-section">
          <ExerciseList 
            exercises={filteredExercises}
            onExerciseSelect={handleExerciseSelect}
          />
        </div>
        
        <div className="performance-section">
          <PerformanceDashboard metrics={performanceMetrics} />
        </div>
      </div>
    </div>
  );
});

FitnessCoach.displayName = 'FitnessCoach';

export default FitnessCoach;