import { Pose } from '@mediapipe/pose';
import { PoseLandmark, PoseResults } from '../types';

// Memoized pose instance to prevent recreation
let poseInstance: Pose | null = null;

export const getPoseInstance = (): Pose => {
  if (!poseInstance) {
    poseInstance = new Pose({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
      }
    });

    poseInstance.setOptions({
      modelComplexity: 1, // Use 0 for faster processing, 1 for better accuracy
      smoothLandmarks: true,
      enableSegmentation: false, // Disable for better performance
      smoothSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
  }
  return poseInstance;
};

// Memoized pose analysis function
const poseAnalysisCache = new Map<string, any>();

export const analyzePose = (results: PoseResults): any => {
  const cacheKey = JSON.stringify(results.poseLandmarks);
  
  if (poseAnalysisCache.has(cacheKey)) {
    return poseAnalysisCache.get(cacheKey);
  }

  const analysis = {
    posture: analyzePosture(results.poseLandmarks),
    movement: analyzeMovement(results.poseLandmarks),
    balance: analyzeBalance(results.poseLandmarks)
  };

  // Cache for 1 second to avoid excessive computation
  poseAnalysisCache.set(cacheKey, analysis);
  setTimeout(() => poseAnalysisCache.delete(cacheKey), 1000);

  return analysis;
};

const analyzePosture = (landmarks: PoseLandmark[]) => {
  if (!landmarks || landmarks.length < 33) return { score: 0, feedback: [] };

  const shoulders = [landmarks[11], landmarks[12]];
  const hips = [landmarks[23], landmarks[24]];
  
  // Calculate shoulder alignment
  const shoulderSlope = Math.abs(shoulders[0].y - shoulders[1].y);
  const hipSlope = Math.abs(hips[0].y - hips[1].y);
  
  const postureScore = Math.max(0, 100 - (shoulderSlope + hipSlope) * 100);
  
  return {
    score: postureScore,
    feedback: postureScore < 70 ? ['Try to keep your shoulders level'] : []
  };
};

const analyzeMovement = (landmarks: PoseLandmark[]) => {
  if (!landmarks || landmarks.length < 33) return { score: 0, feedback: [] };

  // Analyze movement smoothness and range
  const elbows = [landmarks[13], landmarks[14]];
  const knees = [landmarks[25], landmarks[26]];
  
  return {
    score: 85, // Placeholder
    feedback: []
  };
};

const analyzeBalance = (landmarks: PoseLandmark[]) => {
  if (!landmarks || landmarks.length < 33) return { score: 0, feedback: [] };

  // Analyze center of gravity
  const leftAnkle = landmarks[27];
  const rightAnkle = landmarks[28];
  
  const centerX = (leftAnkle.x + rightAnkle.x) / 2;
  const balanceScore = Math.max(0, 100 - Math.abs(centerX - 0.5) * 200);
  
  return {
    score: balanceScore,
    feedback: balanceScore < 70 ? ['Try to distribute weight evenly'] : []
  };
};

// Cleanup function to free memory
export const cleanupPoseDetection = () => {
  if (poseInstance) {
    poseInstance.close();
    poseInstance = null;
  }
  poseAnalysisCache.clear();
};