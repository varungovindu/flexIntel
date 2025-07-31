import React, { memo, useMemo } from 'react';
import { PerformanceMetrics } from '../types';

interface PerformanceDashboardProps {
  metrics: PerformanceMetrics;
}

const PerformanceDashboard: React.FC<PerformanceDashboardProps> = memo(({ metrics }) => {
  // Memoized performance indicators
  const performanceIndicators = useMemo(() => {
    const getFormColor = (form: string) => {
      switch (form) {
        case 'good': return '#4CAF50';
        case 'needs_improvement': return '#FF9800';
        case 'poor': return '#F44336';
        default: return '#9E9E9E';
      }
    };

    const getAccuracyColor = (accuracy: number) => {
      if (accuracy >= 80) return '#4CAF50';
      if (accuracy >= 60) return '#FF9800';
      return '#F44336';
    };

    return {
      formColor: getFormColor(metrics.form),
      accuracyColor: getAccuracyColor(metrics.accuracy)
    };
  }, [metrics.form, metrics.accuracy]);

  // Memoized feedback display
  const feedbackDisplay = useMemo(() => {
    if (metrics.feedback.length === 0) {
      return <p className="no-feedback">Great form! Keep it up!</p>;
    }

    return (
      <ul className="feedback-list">
        {metrics.feedback.map((feedback, index) => (
          <li key={index} className="feedback-item">
            {feedback}
          </li>
        ))}
      </ul>
    );
  }, [metrics.feedback]);

  return (
    <div className="performance-dashboard">
      <h3>Performance Metrics</h3>
      
      <div className="metrics-grid">
        <div className="metric-card">
          <h4>Accuracy</h4>
          <div 
            className="metric-value"
            style={{ color: performanceIndicators.accuracyColor }}
          >
            {Math.round(metrics.accuracy)}%
          </div>
        </div>
        
        <div className="metric-card">
          <h4>Reps</h4>
          <div className="metric-value">
            {metrics.reps}
          </div>
        </div>
        
        <div className="metric-card">
          <h4>Form</h4>
          <div 
            className="metric-value"
            style={{ color: performanceIndicators.formColor }}
          >
            {metrics.form.replace('_', ' ')}
          </div>
        </div>
      </div>
      
      <div className="feedback-section">
        <h4>Real-time Feedback</h4>
        {feedbackDisplay}
      </div>
    </div>
  );
});

PerformanceDashboard.displayName = 'PerformanceDashboard';

export default PerformanceDashboard;