import { useEffect, useRef } from "react";
import {
  FilesetResolver,
  HandLandmarker
} from "@mediapipe/tasks-vision";

export default function HandTracker() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handLandmarkerRef = useRef(null);

  useEffect(() => {
    let animationFrame;
    let stream;
    let isActive = true;
    let lastDetectionTime = 0;
    let lastResults = null;
    let lastSeenTime = 0;

    const detectionInterval = 100;
    const landmarkGracePeriod = 450;
    const trackedLandmarks = [0, 10];

    async function setup() {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
      );

      const detectorOptions = {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        },

        runningMode: "VIDEO",

        numHands: 2,

        minHandDetectionConfidence: 0.3,
        minHandPresenceConfidence: 0.3,
        minTrackingConfidence: 0.3
      };

      let handLandmarker;

      try {
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          ...detectorOptions,
          baseOptions: {
            ...detectorOptions.baseOptions,
            delegate: "GPU"
          }
        });
      } catch {
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          ...detectorOptions,
          baseOptions: {
            ...detectorOptions.baseOptions,
            delegate: "CPU"
          }
        });
      }

      if (!isActive) {
        handLandmarker.close();
        return;
      }

      handLandmarkerRef.current = handLandmarker;

      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 }
        },
        audio: false
      });

      if (!isActive) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      videoRef.current.srcObject = stream;

      await videoRef.current.play();

      detectHands();
    }

    function detectHands() {
      const video = videoRef.current;

      if (!video || !handLandmarkerRef.current) {
        animationFrame = requestAnimationFrame(detectHands);
        return;
      }

      const now = performance.now();

      if (video.readyState >= 2 && now - lastDetectionTime >= detectionInterval) {
        lastDetectionTime = now;
        const results = handLandmarkerRef.current.detectForVideo(video, now);

        if (results.landmarks?.length) {
          lastResults = results;
          lastSeenTime = now;
        }

        drawHands(lastResults, now - lastSeenTime <= landmarkGracePeriod);
      }

      animationFrame = requestAnimationFrame(detectHands);
    }

    function drawHands(results, shouldKeepLastResults) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      const videoWidth = videoRef.current.videoWidth;
      const videoHeight = videoRef.current.videoHeight;

      if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
        canvas.width = videoWidth;
        canvas.height = videoHeight;
      }

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      if (!results?.landmarks || !shouldKeepLastResults) return;

      const handAnchors = results.landmarks.slice(0, 2).map((hand) => {
        const points = trackedLandmarks.map((landmarkIndex) => hand[landmarkIndex]);

        for (const point of points) {
          drawPoint(ctx, point, canvas.width, canvas.height);
        }

        return getPointBetween(points[0], points[1], canvas.width, canvas.height);
      });

      if (handAnchors.length === 2) {
        drawDistanceOverlay(ctx, handAnchors[0], handAnchors[1], canvas.width, canvas.height);
      }
    }

    function getPointBetween(firstPoint, secondPoint, width, height) {
      return {
        x: ((firstPoint.x + secondPoint.x) / 2) * width,
        y: ((firstPoint.y + secondPoint.y) / 2) * height
      };
    }

    function drawPoint(ctx, point, width, height) {
      const x = point.x * width;
      const y = point.y * height;

      ctx.beginPath();
      ctx.arc(x, y, Math.max(7, width / 90), 0, Math.PI * 2);
      ctx.fillStyle = "#f7f4ff";
      ctx.fill();
      ctx.lineWidth = Math.max(3, width / 240);
      ctx.strokeStyle = "#ff3d81";
      ctx.stroke();
    }

    function drawDistanceOverlay(ctx, firstAnchor, secondAnchor, width, height) {
      const deltaX = secondAnchor.x - firstAnchor.x;
      const deltaY = secondAnchor.y - firstAnchor.y;
      const distance = Math.sqrt(deltaX ** 2 + deltaY ** 2);
      const midpoint = {
        x: (firstAnchor.x + secondAnchor.x) / 2,
        y: (firstAnchor.y + secondAnchor.y) / 2
      };

      ctx.save();
      ctx.lineWidth = Math.max(4, width / 180);
      ctx.setLineDash([width / 55, width / 90]);
      ctx.strokeStyle = "#7df9ff";
      ctx.shadowColor = "#7df9ff";
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.moveTo(firstAnchor.x, firstAnchor.y);
      ctx.lineTo(secondAnchor.x, secondAnchor.y);
      ctx.stroke();
      ctx.restore();

      drawLightstick(ctx, midpoint.x, midpoint.y, deltaX, deltaY, width);

      const distanceLabel = `${Math.round(distance)} px  /  ${Math.round(distance / width * 100)}%`;
      const labelX = midpoint.x;
      const labelY = Math.max(34, midpoint.y - height / 12);
      const fontSize = Math.max(16, width / 38);

      ctx.font = `600 ${fontSize}px ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const labelWidth = ctx.measureText(distanceLabel).width + fontSize * 1.4;

      ctx.fillStyle = "rgba(8, 12, 28, 0.86)";
      roundRect(ctx, labelX - labelWidth / 2, labelY - fontSize, labelWidth, fontSize * 2, fontSize / 2);
      ctx.fill();
      ctx.fillStyle = "#f7f4ff";
      ctx.fillText(distanceLabel, labelX, labelY);
    }

    function drawLightstick(ctx, x, y, deltaX, deltaY, width) {
      const angle = Math.atan2(deltaY, deltaX);
      const length = Math.min(width * 0.28, Math.max(width * 0.16, Math.hypot(deltaX, deltaY) * 0.72));
      const handleLength = length * 0.28;

      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(angle);
      ctx.shadowColor = "#ff3d81";
      ctx.shadowBlur = 24;
      ctx.fillStyle = "#ff3d81";
      roundRect(ctx, -length / 2, -Math.max(7, width / 100), length, Math.max(14, width / 50), width / 100);
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#f7f4ff";
      roundRect(ctx, -length / 2 + handleLength, -Math.max(4, width / 180), length - handleLength, Math.max(8, width / 90), width / 180);
      ctx.fillStyle = "#25243a";
      roundRect(ctx, -length / 2, -Math.max(9, width / 85), handleLength, Math.max(18, width / 42), width / 100);
      ctx.restore();
    }

    function roundRect(ctx, x, y, width, height, radius) {
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, radius);
    }

    setup();

    return () => {
      isActive = false;
      cancelAnimationFrame(animationFrame);
      stream?.getTracks().forEach((track) => track.stop());
      handLandmarkerRef.current?.close();
    };
  }, []);

  return (
    <div className="hand-tracker">
      <video
        ref={videoRef}
        playsInline
        muted
      />

      <canvas
        ref={canvasRef}
      />

      <div className="hand-tracker__legend">2-point lightstick tracking</div>
    </div>
  );
}