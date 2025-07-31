import React, { useState, useEffect, useCallback, memo } from 'react';
import { Exercise, PerformanceMetrics } from '../types';
import PoseDetection from './PoseDetection';

interface WorkoutSessionProps {
  exercise: Exercise;
  poseDetection: any;
  performanceMetrics: PerformanceMetrics;
  onEnd: () => void;
}

const WorkoutSession: React.FC<WorkoutSessionProps> = memo(({ 
  exercise, 
  poseDetection, 
  performanceMetrics, 
  onEnd 
}) => {
  const [sessionTime, setSessionTime] = useState(0);
  const [currentInstruction, setCurrentInstruction] = useState(0);
  const [isPaused, setIsPaused] = useState(false);

  // Timer effect
  useEffect(() => {
    if (isPaused) return;

    const timer = setInterval(() => {
      setSessionTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [isPaused]);

  // Auto-advance instructions based on performance
  useEffect(() => {
    if (performanceMetrics.accuracy > 80 && currentInstruction < exercise.instructions.length - 1) {
      const timer = setTimeout(() => {
        setCurrentInstruction(prev => prev + 1);
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [performanceMetrics.accuracy, currentInstruction, exercise.instructions.length]);

  const handlePause = useCallback(() => {
    setIsPaused(prev => !prev);
  }, []);

  const handleEnd = useCallback(() => {
    onEnd();
  }, [onEnd]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="workout-session">
      <div className="session-header">
        <h2>{exercise.name}</h2>
        <div className="session-controls">
          <button onClick={handlePause} className="control-btn">
            {isPaused ? 'Resume' : 'Pause'}
          </button>
          <button onClick={handleEnd} className="control-btn end-btn">
            End Workout
          </button>
        </div>
      </div>

      <div className="session-content">
        <div className="pose-section">
          <PoseDetection poseDetection={poseDetection} />
        </div>

        <div className="instruction-section">
          <h3>Current Instruction</h3>
          <div className="instruction-display">
            <p className="instruction-text">
              {exercise.instructions[currentInstruction]}
            </p>
            <div className="instruction-progress">
              {currentInstruction + 1} of {exercise.instructions.length}
            </div>
          </div>

          <div className="session-stats">
            <div className="stat-item">
              <span className="stat-label">Time:</span>
              <span className="stat-value">{formatTime(sessionTime)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Accuracy:</span>
              <span className="stat-value">{Math.round(performanceMetrics.accuracy)}%</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Form:</span>
              <span className={`stat-value form-${performanceMetrics.form}`}>
                {performanceMetrics.form.replace('_', ' ')}
              </span>
            </div>
          </div>

          {performanceMetrics.feedback.length > 0 && (
            <div className="real-time-feedback">
              <h4>Feedback</h4>
              <ul>
                {performanceMetrics.feedback.map((feedback, index) => (
                  <li key={index} className="feedback-item">
                    {feedback}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

WorkoutSession.displayName = 'WorkoutSession';

export default WorkoutSession;