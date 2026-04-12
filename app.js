import {
  FilesetResolver,
  PoseLandmarker,
} from "https://unpkg.com/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs";

const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const WASM_ROOT =
  "https://unpkg.com/@mediapipe/tasks-vision@0.10.32/wasm";

const LANDMARK = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
};

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const overlayCanvas = document.getElementById("overlayCanvas");
const emptyPreview = document.getElementById("emptyPreview");
const statusPill = document.getElementById("statusPill");
const errorBox = document.getElementById("errorBox");
const resultList = document.getElementById("resultList");
const totalPeople = document.getElementById("totalPeople");
const raisedPeople = document.getElementById("raisedPeople");
const notRaisedPeople = document.getElementById("notRaisedPeople");
const raisedPercent = document.getElementById("raisedPercent");
const notRaisedPercent = document.getElementById("notRaisedPercent");
const raisedBar = document.getElementById("raisedBar");
const notRaisedBar = document.getElementById("notRaisedBar");
const uploadButton = document.querySelector(".upload-button");

let detector = null;
let currentObjectUrl = null;
const sourceCanvas = document.createElement("canvas");

function extractPoseLandmarks(detection) {
  if (!detection) return [];
  if (Array.isArray(detection.landmarks)) return detection.landmarks;
  if (Array.isArray(detection.poseLandmarks)) return detection.poseLandmarks;
  if (detection.result && Array.isArray(detection.result.landmarks)) {
    return detection.result.landmarks;
  }
  return [];
}

function setStatus(message, ready = false) {
  statusPill.textContent = message;
  statusPill.classList.toggle("ready", ready);
}

function setError(message = "") {
  errorBox.textContent = message;
  errorBox.classList.toggle("hidden", !message);
}

function toPercent(count, total) {
  if (!total) return 0;
  return Number(((count / total) * 100).toFixed(1));
}

function isReliable(point) {
  return point && (point.visibility ?? 1) >= 0.4;
}

function isHandRaised(landmarks, side) {
  const wrist =
    side === "left" ? landmarks[LANDMARK.leftWrist] : landmarks[LANDMARK.rightWrist];
  const elbow =
    side === "left" ? landmarks[LANDMARK.leftElbow] : landmarks[LANDMARK.rightElbow];
  const shoulder =
    side === "left"
      ? landmarks[LANDMARK.leftShoulder]
      : landmarks[LANDMARK.rightShoulder];
  const nose = landmarks[LANDMARK.nose];

  if (!isReliable(wrist) || !isReliable(elbow) || !isReliable(shoulder)) {
    return false;
  }

  const wristAboveShoulder = wrist.y < shoulder.y - 0.08;
  const elbowHighEnough = elbow.y < shoulder.y + 0.02;
  const wristNearBody = Math.abs(wrist.x - shoulder.x) < 0.35;
  const wristAboveFace = isReliable(nose) ? wrist.y < nose.y + 0.02 : false;

  return wristNearBody && wristAboveShoulder && (elbowHighEnough || wristAboveFace);
}

function getBounds(landmarks, width, height) {
  const xs = landmarks.map((point) => point.x * width);
  const ys = landmarks.map((point) => point.y * height);

  const minX = Math.max(Math.min(...xs) - 24, 0);
  const maxX = Math.min(Math.max(...xs) + 24, width);
  const minY = Math.max(Math.min(...ys) - 24, 0);
  const maxY = Math.min(Math.max(...ys) + 24, height);

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

function analyzePoses(poseLandmarks, width, height) {
  const people = poseLandmarks.map((landmarks, index) => {
    const raised = isHandRaised(landmarks, "left") || isHandRaised(landmarks, "right");

    return {
      id: index + 1,
      raised,
      label: raised ? "Hand Raised" : "Hand Not Raised",
      bounds: getBounds(landmarks, width, height),
    };
  });

  const raisedCount = people.filter((person) => person.raised).length;
  const totalCount = people.length;
  const notRaisedCount = totalCount - raisedCount;

  return {
    totalCount,
    raisedCount,
    notRaisedCount,
    raisedPct: toPercent(raisedCount, totalCount),
    notRaisedPct: toPercent(notRaisedCount, totalCount),
    people,
  };
}

function updateSummary(result) {
  totalPeople.textContent = result.totalCount;
  raisedPeople.textContent = result.raisedCount;
  notRaisedPeople.textContent = result.notRaisedCount;
  raisedPercent.textContent = `${result.raisedPct}%`;
  notRaisedPercent.textContent = `${result.notRaisedPct}%`;
  raisedBar.style.width = `${result.raisedPct}%`;
  notRaisedBar.style.width = `${result.notRaisedPct}%`;
}

function updateResults(result) {
  if (!result.people.length) {
    resultList.innerHTML =
      '<p class="empty-results">No people were detected. Try a clearer image with visible upper bodies.</p>';
    return;
  }

  resultList.innerHTML = result.people
    .map(
      (person) => `
        <div class="result-row ${person.raised ? "raised" : "not-raised"}">
          <span>Person ${person.id}</span>
          <strong>${person.label}</strong>
        </div>
      `
    )
    .join("");
}

function drawOverlay(result) {
  const context = overlayCanvas.getContext("2d");
  if (!context) return;

  overlayCanvas.width = previewImage.naturalWidth;
  overlayCanvas.height = previewImage.naturalHeight;
  context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  result.people.forEach((person) => {
    const { x, y, width, height } = person.bounds;
    context.strokeStyle = person.raised ? "#149f6d" : "#b65044";
    context.lineWidth = 4;
    context.strokeRect(x, y, width, height);

    context.fillStyle = person.raised ? "#149f6d" : "#b65044";
    context.fillRect(x, Math.max(y - 34, 6), Math.max(width * 0.44, 158), 28);

    context.fillStyle = "#fffaf2";
    context.font = "600 16px Arial";
    context.fillText(`Person ${person.id}: ${person.label}`, x + 10, Math.max(y - 15, 24));
  });
}

async function initDetector() {
  try {
    const resolver = await FilesetResolver.forVisionTasks(WASM_ROOT);
    detector = await PoseLandmarker.createFromOptions(resolver, {
      baseOptions: { modelAssetPath: MODEL_ASSET_PATH },
      runningMode: "IMAGE",
      numPoses: 10,
    });

    imageInput.disabled = false;
    uploadButton.classList.remove("disabled");
    setStatus("Model ready. Upload an image to detect raised hands.", true);
  } catch (error) {
    console.error(error);
    setError("Could not load the pose model. Keep internet on for the first load, then refresh.");
    setStatus("Model unavailable");
  }
}

async function analyzeCurrentImage() {
  if (!detector) return;

  setError("");
  setStatus("Analyzing image...");

  try {
    sourceCanvas.width = previewImage.naturalWidth;
    sourceCanvas.height = previewImage.naturalHeight;
    const sourceContext = sourceCanvas.getContext("2d");
    sourceContext.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    sourceContext.drawImage(previewImage, 0, 0);

    const detection = detector.detect(sourceCanvas);
    const poseLandmarks = extractPoseLandmarks(detection);
    console.log("Pose detection result:", detection);

    const result = analyzePoses(
      poseLandmarks,
      previewImage.naturalWidth,
      previewImage.naturalHeight
    );

    updateSummary(result);
    updateResults(result);
    drawOverlay(result);
    overlayCanvas.classList.remove("hidden");

    setStatus(
      result.totalCount
        ? `Detected ${result.totalCount} people in the image.`
        : "No people were detected by the pose model. Try a clearer image with full upper bodies.",
      true
    );
  } catch (error) {
    console.error(error);
    setError("Image analysis failed. Try another image with people visible from head to hands.");
    setStatus("Analysis failed");
  }
}

imageInput.disabled = true;
uploadButton.classList.add("disabled");

imageInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
  }

  currentObjectUrl = URL.createObjectURL(file);
  previewImage.src = currentObjectUrl;
  previewImage.classList.remove("hidden");
  emptyPreview.classList.add("hidden");
  overlayCanvas.classList.add("hidden");
  setStatus("Image loaded. Running hand-raise detection...", true);
});

previewImage.addEventListener("load", () => {
  void analyzeCurrentImage();
});

window.addEventListener("beforeunload", () => {
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
  }
  detector?.close();
});

void initDetector();
