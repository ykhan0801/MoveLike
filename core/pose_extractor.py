# Pose extraction via MediaPipe Tasks API (0.10+).
# The pose_landmarker_lite model (~6 MB) is fetched on first run.

import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# Joints we care about, as (a, b, c) landmark indices where the angle is at b.
# Indices follow MediaPipe's 33-landmark pose schema.
JOINT_CONNECTIONS = {
    "left_elbow":     (11, 13, 15),  # shoulder → elbow → wrist
    "right_elbow":    (12, 14, 16),
    "left_knee":      (23, 25, 27),  # hip → knee → ankle
    "right_knee":     (24, 26, 28),
    "left_hip":       (11, 23, 25),  # shoulder → hip → knee
    "right_hip":      (12, 24, 26),
    "left_shoulder":  (13, 11, 23),  # elbow → shoulder → hip
    "right_shoulder": (14, 12, 24),
}

# Skeleton edges used when drawing the overlay.
DRAW_CONNECTIONS = [
    (0, 11), (0, 12),      # nose to shoulders
    (11, 12),              # shoulder bar
    (11, 13), (13, 15),    # left arm
    (12, 14), (14, 16),    # right arm
    (11, 23), (12, 24),    # torso
    (23, 24),              # hip bar
    (23, 25), (25, 27),    # left leg
    (24, 26), (26, 28),    # right leg
    (27, 29), (28, 30),    # ankle to heel
]


@dataclass
class FramePose:
    frame_idx: int
    landmarks: Optional[np.ndarray]   # (33, 4): x, y, z, visibility
    joint_angles: dict = field(default_factory=dict)


def ensure_model(progress_callback=None) -> str:
    # Download the .task file on first use; no-op otherwise.
    os.makedirs(_MODEL_DIR, exist_ok=True)

    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 1_000_000:
        return _MODEL_PATH

    if progress_callback:
        progress_callback("Downloading pose model (first run only, ~6 MB)...")

    def _reporthook(count, block_size, total_size):
        if progress_callback and total_size > 0:
            pct = min(100, int(count * block_size / total_size * 100))
            progress_callback(f"Downloading pose model... {pct}%")

    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH, reporthook=_reporthook)
    except Exception as e:
        raise RuntimeError(
            f"Could not download pose model: {e}\n\n"
            "Please manually download it from:\n"
            f"  {_MODEL_URL}\n"
            f"and save to:\n  {os.path.abspath(_MODEL_PATH)}"
        )

    if os.path.getsize(_MODEL_PATH) < 1_000_000:
        os.remove(_MODEL_PATH)
        raise RuntimeError("Downloaded model file appears corrupt. Please retry.")

    return _MODEL_PATH


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # Angle in degrees at vertex b, using only the x/y components.
    a2, b2, c2 = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a2 - b2, c2 - b2
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def extract_poses(
    video_path: str,
    max_frames: int = 300,
    progress_callback=None,
) -> tuple[list[FramePose], float, int]:
    # Pull pose landmarks from a video, uniformly sampling up to max_frames
    # so longer clips don't blow up runtime. Returns (poses, fps, total_frames).
    model_path = ensure_model(progress_callback)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_every = max(1, total_frames // max_frames)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    poses: list[FramePose] = []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # VIDEO mode wants strictly increasing timestamps (ms).
                timestamp_ms = int(frame_idx / fps * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    raw_lms = result.pose_landmarks[0]
                    lm_array = np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility or 0.0] for lm in raw_lms],
                        dtype=np.float32,
                    )

                    angles = {
                        joint: calculate_angle(
                            lm_array[a_i, :3],
                            lm_array[b_i, :3],
                            lm_array[c_i, :3],
                        )
                        for joint, (a_i, b_i, c_i) in JOINT_CONNECTIONS.items()
                    }

                    poses.append(FramePose(
                        frame_idx=frame_idx,
                        landmarks=lm_array,
                        joint_angles=angles,
                    ))
                else:
                    poses.append(FramePose(frame_idx=frame_idx, landmarks=None))

            frame_idx += 1

    cap.release()
    return poses, fps, total_frames


def poses_to_angle_timeseries(poses: list[FramePose]) -> dict:
    # Flatten a list of FramePose into {joint: [angle or None per frame]}.
    series = {j: [] for j in JOINT_CONNECTIONS}
    for fp in poses:
        for j in JOINT_CONNECTIONS:
            series[j].append(fp.joint_angles.get(j) if fp.joint_angles else None)
    return series
