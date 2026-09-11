import { useEffect, useRef } from "react";
import {
  FilesetResolver,
  HandLandmarker
} from "@mediapipe/tasks-vision";
import tumblerUrl from "./assets/tumbler.png";

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

    const detectionInterval = 50;

    const landmarkGracePeriod = 450;
    // 0 is base bottom of the palm and 10 is the top most of the plam's upper part...
    // const trackedLandmarks = [10, 0];
    const trackedLandmarks = [5, 17];
    const tumblerImage = new Image();
    tumblerImage.src = tumblerUrl;
    tumblerImage.onload = () => {
      if (isActive && lastResults) {
        drawHands(lastResults, performance.now() - lastSeenTime <= landmarkGracePeriod);
      }
    };

    async function setup() {
      const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm");
      const detectorOptions = {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        },
        runningMode: "VIDEO",
        numHands: 2,
        // Hand tollerances
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
            ...detectorOptions, baseOptions: { ...detectorOptions.baseOptions, delegate: "CPU"
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
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 60, max: 60 }
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
      if (!video || !handLandmarkerRef.current) { animationFrame = requestAnimationFrame(detectHands); return; }

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

      const videoWidth = videoRef.current.videoWidth, videoHeight = videoRef.current.videoHeight;

      if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
        canvas.width = videoWidth;
        canvas.height = videoHeight;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!results?.landmarks || !shouldKeepLastResults) return;

      const handAnchors = results.landmarks.slice(0, 2).map((hand) => {
        const points = trackedLandmarks.map((landmarkIndex) => hand[landmarkIndex]);
        for (const point of points) drawPoint(ctx, point, canvas.width, canvas.height);
        return {
          anchor: getPointBetween(points[0], points[1], canvas.width, canvas.height),
          firstPoint: points[0],
          secondPoint: points[1]
        };
      });

      for (const hand of handAnchors) {
        drawChayaGlass(ctx, hand, canvas.width, canvas.height);
      }

      if (handAnchors.length === 2) {
        drawDistanceOverlay(ctx, handAnchors[0].anchor, handAnchors[1].anchor, canvas.width, canvas.height);
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
      ctx.fillStyle = "#f00";
      ctx.fill();
      ctx.lineWidth = Math.max(3, width / 40);
    }

    function drawDistanceOverlay(ctx, firstAnchor, secondAnchor, width, height) {
      const deltaX = secondAnchor.x - firstAnchor.x;
      const deltaY = secondAnchor.y - firstAnchor.y;
      
      const distance = Math.sqrt(deltaX ** 2 + deltaY ** 2);
      const midpoint = {
        x: (firstAnchor.x + secondAnchor.x) / 2,
        y: (firstAnchor.y + secondAnchor.y) / 2
      };


      const distanceLabel = `${Math.round(distance)}`;
      const labelX = midpoint.x;
      const labelY = Math.max(34, midpoint.y - height / 12);
      const fontSize = Math.max(16, width / 38);

      ctx.font = `600 ${fontSize}px ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const labelWidth = ctx.measureText(distanceLabel).width + fontSize * 1.4;

      ctx.fillStyle = "#fff3";
      roundRect(ctx, labelX - labelWidth / 2, labelY - fontSize, labelWidth, fontSize * 2, fontSize / 2);
      ctx.fill();
      ctx.fillText(distanceLabel, labelX, labelY);
    }

    function drawChayaGlass(ctx, hand, width, height) {
      if (!tumblerImage.complete || !tumblerImage.naturalWidth) return;

      const firstPoint = {
        x: hand.firstPoint.x * width,
        y: hand.firstPoint.y * height
      };
      const secondPoint = {
        x: hand.secondPoint.x * width,
        y: hand.secondPoint.y * height
      };
      const handLength = Math.hypot(secondPoint.x - firstPoint.x, secondPoint.y - firstPoint.y);
      const glassSize = Math.max(110, Math.min(handLength * 3.2, width * 0.3));
      const angle = Math.atan2(secondPoint.y - firstPoint.y, secondPoint.x - firstPoint.x) - Math.PI / 2;

      ctx.save();
      ctx.translate(hand.anchor.x, hand.anchor.y);
      ctx.rotate(angle);
      ctx.shadowColor = "rgba(0, 0, 0, 0.35)";
      ctx.shadowBlur = Math.max(8, width / 90);
      ctx.drawImage(tumblerImage, -glassSize / 2, -glassSize / 2, glassSize, glassSize);
      ctx.restore();
    }

    // function drawLightstick(ctx, x, y, deltaX, deltaY, width) {
    //   const angle = Math.atan2(deltaY, deltaX);
    //   const length = Math.min(width * 0.28, Math.max(width * 0.16, Math.hypot(deltaX, deltaY) * 0.72));
    //   const handleLength = length * 0.28;

    //   ctx.save();
    //   ctx.translate(x, y);
    //   ctx.rotate(angle);
    //   ctx.shadowColor = "#ff3d81";
    //   ctx.shadowBlur = 24;
    //   ctx.fillStyle = "#ff3d81";
    //   roundRect(ctx, -length / 2, -Math.max(7, width / 100), length, Math.max(14, width / 50), width / 100);
    //   ctx.shadowBlur = 0;
    //   ctx.fillStyle = "#f7f4ff";
    //   roundRect(ctx, -length / 2 + handleLength, -Math.max(4, width / 180), length - handleLength, Math.max(8, width / 90), width / 180);
    //   ctx.fillStyle = "#25243a";
    //   roundRect(ctx, -length / 2, -Math.max(9, width / 85), handleLength, Math.max(18, width / 42), width / 100);
    //   ctx.restore();
    // }

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

    </div>
  );
}