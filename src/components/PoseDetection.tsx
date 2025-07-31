import React, { useRef, useEffect, useCallback, memo } from 'react';
import Webcam from 'react-webcam';
import { usePoseDetection } from '../hooks/usePoseDetection';

interface PoseDetectionProps {
  poseDetection: ReturnType<typeof usePoseDetection>;
}

const PoseDetection: React.FC<PoseDetectionProps> = memo(({ poseDetection }) => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Optimized frame processing
  const processFrame = useCallback(() => {
    if (webcamRef.current && poseDetection.isInitialized) {
      const video = webcamRef.current.video;
      if (video && video.readyState === video.HAVE_ENOUGH_DATA) {
        poseDetection.sendFrame(video);
      }
    }
    animationFrameRef.current = requestAnimationFrame(processFrame);
  }, [poseDetection]);

  // Start frame processing
  useEffect(() => {
    if (poseDetection.isInitialized) {
      processFrame();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [poseDetection.isInitialized, processFrame]);

  // Optimized canvas rendering
  const drawPoseLandmarks = useCallback((landmarks: any[]) => {
    const canvas = canvasRef.current;
    if (!canvas || !landmarks) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas efficiently
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw landmarks
    landmarks.forEach((landmark: any) => {
      if (landmark.visibility > 0.5) {
        ctx.beginPath();
        ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 3, 0, 2 * Math.PI);
        ctx.fillStyle = '#00ff00';
        ctx.fill();
      }
    });
  }, []);

  // Update canvas when pose results change
  useEffect(() => {
    if (poseDetection.lastResults?.poseLandmarks) {
      drawPoseLandmarks(poseDetection.lastResults.poseLandmarks);
    }
  }, [poseDetection.lastResults, drawPoseLandmarks]);

  return (
    <div className="pose-detection">
      <div className="camera-container">
        <Webcam
          ref={webcamRef}
          audio={false}
          width={640}
          height={480}
          style={{ display: 'block' }}
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            pointerEvents: 'none'
          }}
        />
      </div>
      
      <div className="pose-status">
        {poseDetection.error && (
          <div className="error-message">
            Error: {poseDetection.error}
          </div>
        )}
        {poseDetection.isProcessing && (
          <div className="processing-indicator">
            Processing pose...
          </div>
        )}
        {poseDetection.isInitialized && (
          <div className="status-message">
            Pose detection active
          </div>
        )}
      </div>
    </div>
  );
});

PoseDetection.displayName = 'PoseDetection';

export default PoseDetection;