export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface PoseResults {
  poseLandmarks: PoseLandmark[];
  poseWorldLandmarks: PoseLandmark[];
}

export interface Exercise {
  id: string;
  name: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  category: 'strength' | 'cardio' | 'flexibility' | 'balance';
  instructions: string[];
  targetMuscles: string[];
}

export interface WorkoutSession {
  id: string;
  startTime: Date;
  endTime?: Date;
  exercises: Exercise[];
  duration: number;
  caloriesBurned?: number;
}

export interface PerformanceMetrics {
  accuracy: number;
  reps: number;
  form: 'good' | 'needs_improvement' | 'poor';
  feedback: string[];
}