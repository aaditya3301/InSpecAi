import base64
import threading

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from facenet_pytorch import MTCNN
from .model import pred_func


MODEL_NAME = "cvit2"
WEIGHT = "cvit2_deepfake_detection_ep_50.pth"
SAMPLED_FRAMES = 15

device = torch.device(pred_func.device)
_model = None
_mtcnn = None
_model_lock = threading.Lock()
_mtcnn_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                loaded = pred_func.load_cvit(WEIGHT, MODEL_NAME, fp16=False)
                loaded.eval()
                _model = loaded
    return _model


def get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        with _mtcnn_lock:
            if _mtcnn is None:
                _mtcnn = MTCNN(keep_all=True, device=device)
    return _mtcnn


def encode_frame(frame, max_width=180):
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return ""

    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_frames(video_path, num_frames=SAMPLED_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    step = max(total_frames // num_frames, 1)

    for i in range(num_frames):
        frame_number = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_number / fps) * 1000) if fps else 0
        frames.append(
            {
                "frame_number": int(frame_number),
                "timestamp_ms": timestamp_ms,
                "image": frame,
                "thumbnail": encode_frame(frame),
            }
        )

    cap.release()
    return frames


def extract_faces(frame):
    faces = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = get_mtcnn().detect(rgb)

    if boxes is None:
        return faces

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        face = rgb[y1:y2, x1:x2]
        if face.size > 0:
            faces.append(cv2.resize(face, (224, 224)))

    return faces


def faces_to_tensor(faces):
    faces = np.array(faces) / 255.0
    faces = np.transpose(faces, (0, 3, 1, 2))
    faces = torch.tensor(faces, dtype=torch.float32)
    faces = F.normalize(faces, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return faces.to(device)


def logits_to_result(logits):
    probs = torch.softmax(logits, dim=0)
    class_idx = torch.argmax(probs).item()
    prediction = "FAKE" if class_idx == 0 else "REAL"
    confidence = probs[class_idx].item()
    return prediction, confidence


def predict_single(video_path):
    frames = extract_frames(video_path)
    frame_results = []
    frame_logits = []

    for index, frame_data in enumerate(frames, start=1):
        faces = extract_faces(frame_data["image"])

        if len(faces) == 0:
            frame_results.append(
                {
                    "index": index,
                    "frame_number": frame_data["frame_number"],
                    "timestamp_ms": frame_data["timestamp_ms"],
                    "prediction": "NO_FACE",
                    "confidence": 0.0,
                    "face_count": 0,
                    "thumbnail": frame_data["thumbnail"],
                }
            )
            continue

        face_tensor = faces_to_tensor(faces)
        with torch.no_grad():
            outputs = get_model()(face_tensor)
            mean_logits = outputs.mean(dim=0)
            prediction, confidence = logits_to_result(mean_logits)
            frame_logits.append(mean_logits)

        frame_results.append(
            {
                "index": index,
                "frame_number": frame_data["frame_number"],
                "timestamp_ms": frame_data["timestamp_ms"],
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "face_count": len(faces),
                "thumbnail": frame_data["thumbnail"],
            }
        )

    if len(frame_logits) == 0:
        return {
            "prediction": "NO_FACE",
            "confidence": 0.0,
            "frames": frame_results,
        }

    overall_logits = torch.stack(frame_logits).mean(dim=0)
    prediction, confidence = logits_to_result(overall_logits)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "frames": frame_results,
    }
