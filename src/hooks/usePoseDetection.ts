import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { PoseResults } from '../types';
import { getPoseInstance, analyzePose, cleanupPoseDetection } from '../utils/poseDetection';

interface UsePoseDetectionProps {
  onResults?: (results: PoseResults) => void;
  onAnalysis?: (analysis: any) => void;
  debounceMs?: number;
}

export const usePoseDetection = ({ 
  onResults, 
  onAnalysis, 
  debounceMs = 100 
}: UsePoseDetectionProps = {}) => {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const poseRef = useRef<any>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastResultsRef = useRef<PoseResults | null>(null);

  // Memoized pose instance
  const pose = useMemo(() => {
    try {
      return getPoseInstance();
    } catch (err) {
      setError('Failed to initialize pose detection');
      return null;
    }
  }, []);

  // Debounced analysis function
  const debouncedAnalysis = useCallback((results: PoseResults) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      const analysis = analyzePose(results);
      onAnalysis?.(analysis);
    }, debounceMs);
  }, [onAnalysis, debounceMs]);

  // Process pose results
  const processResults = useCallback((results: PoseResults) => {
    if (!results.poseLandmarks || results.poseLandmarks.length === 0) {
      return;
    }

    lastResultsRef.current = results;
    onResults?.(results);
    debouncedAnalysis(results);
  }, [onResults, debouncedAnalysis]);

  // Initialize pose detection
  useEffect(() => {
    if (!pose) return;

    const initializePose = async () => {
      try {
        setIsProcessing(true);
        
        pose.onResults((results: PoseResults) => {
          processResults(results);
        });

        setIsInitialized(true);
        setError(null);
      } catch (err) {
        setError('Failed to initialize pose detection');
      } finally {
        setIsProcessing(false);
      }
    };

    initializePose();

    // Cleanup on unmount
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      cleanupPoseDetection();
    };
  }, [pose, processResults]);

  // Send frame to pose detection
  const sendFrame = useCallback((imageElement: HTMLImageElement | HTMLVideoElement) => {
    if (!pose || !isInitialized || isProcessing) return;

    try {
      setIsProcessing(true);
      pose.send({ image: imageElement });
    } catch (err) {
      setError('Failed to process frame');
    } finally {
      setIsProcessing(false);
    }
  }, [pose, isInitialized, isProcessing]);

  return {
    isInitialized,
    isProcessing,
    error,
    sendFrame,
    lastResults: lastResultsRef.current
  };
};