# Renders a side-by-side comparison video with skeleton overlays:
# pro on the left (green), user on the right (orange). Doesn't depend
# on mediapipe.solutions so it works with the Tasks API.

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
from moviepy import VideoFileClip

from core.pose_extractor import FramePose, DRAW_CONNECTIONS

PRO_BONE_COLOUR  = (50, 220, 120)
PRO_DOT_COLOUR   = (100, 255, 160)
USER_BONE_COLOUR = (40, 140, 255)
USER_DOT_COLOUR  = (80, 180, 255)
WHITE            = (255, 255, 255)
BLACK            = (0, 0, 0)


def _draw_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    bone_colour: tuple,
    dot_colour: tuple,
    min_visibility: float = 0.4,
) -> np.ndarray:
    # Draws bones and joints directly onto frame.
    if landmarks is None:
        return frame

    h, w = frame.shape[:2]

    def pt(idx):
        return (int(landmarks[idx, 0] * w), int(landmarks[idx, 1] * h))

    def visible(idx):
        return landmarks[idx, 3] >= min_visibility

    for a, b in DRAW_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks) and visible(a) and visible(b):
            cv2.line(frame, pt(a), pt(b), bone_colour, 2, cv2.LINE_AA)

    for i in range(len(landmarks)):
        if visible(i):
            cv2.circle(frame, pt(i), 5, dot_colour, -1, cv2.LINE_AA)
            cv2.circle(frame, pt(i), 5, WHITE, 1, cv2.LINE_AA)

    return frame


def _label(frame: np.ndarray, text: str, colour: tuple) -> np.ndarray:
    # Bold label in the top-left with a black outline for contrast.
    cv2.putText(frame, text, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, BLACK, 5, cv2.LINE_AA)
    cv2.putText(frame, text, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA)
    return frame


def render_comparison_video(
    pro_video_path: str,
    user_video_path: str,
    aligned_pro: list[FramePose],
    aligned_user: list[FramePose],
    output_path: str,
    target_height: int = 480,
) -> str:
    # Writes a side-by-side mp4 with skeletons drawn on both panels.
    # Each panel is square-ish (target_height tall), so the full output is
    # (target_height * 2) wide. Source frames are uniformly resampled so the
    # two clips play at the same pace regardless of their original length.
    cap_pro  = cv2.VideoCapture(pro_video_path)
    cap_user = cv2.VideoCapture(user_video_path)

    fps       = cap_pro.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames  = len(aligned_pro)
    panel_w   = target_height
    out_w     = panel_w * 2
    out_h     = target_height

    # cv2 writes mp4v which most browsers can't play inside an <video> tag.
    # Render to a temp file first, then transcode to H.264 at the end.
    tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_raw.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_raw.name, fourcc, fps, (out_w, out_h))

    total_pro  = int(cap_pro.get(cv2.CAP_PROP_FRAME_COUNT))
    total_user = int(cap_user.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use the original-video frame indices that the aligned poses were
    # sampled from. These reflect the onset trim and DTW pairing, so the
    # two panels stay in sync with the matched poses rather than just
    # scrubbing both clips end to end.
    pro_indices = np.clip(
        np.array([fp.frame_idx for fp in aligned_pro], dtype=int),
        0, max(0, total_pro - 1),
    )
    user_indices = np.clip(
        np.array([fp.frame_idx for fp in aligned_user], dtype=int),
        0, max(0, total_user - 1),
    )

    def read_frames_sequential(cap, indices: np.ndarray) -> dict:
        # Walk the video forward once and only decode the frames we need.
        # Much faster than seeking for every target frame.
        frames: dict[int, np.ndarray] = {}
        current = 0
        for target in sorted(set(indices.tolist())):
            while current < target:
                cap.grab()
                current += 1
            ok, frame = cap.read()
            if ok:
                frames[target] = frame
            current += 1
        return frames

    pro_frames  = read_frames_sequential(cap_pro,  pro_indices)
    user_frames = read_frames_sequential(cap_user, user_indices)

    blank = np.zeros((out_h, panel_w, 3), dtype=np.uint8)

    for i, (fp_pro, fp_user) in enumerate(zip(aligned_pro, aligned_user)):
        pro_frame  = pro_frames.get(pro_indices[i])
        user_frame = user_frames.get(user_indices[i])

        pro_panel  = cv2.resize(pro_frame,  (panel_w, out_h)) if pro_frame  is not None else blank.copy()
        user_panel = cv2.resize(user_frame, (panel_w, out_h)) if user_frame is not None else blank.copy()

        _draw_skeleton(pro_panel,  fp_pro.landmarks,  PRO_BONE_COLOUR,  PRO_DOT_COLOUR)
        _draw_skeleton(user_panel, fp_user.landmarks, USER_BONE_COLOUR, USER_DOT_COLOUR)

        _label(pro_panel,  "PRO", PRO_BONE_COLOUR)
        _label(user_panel, "YOU", USER_BONE_COLOUR)

        combined = np.hstack([pro_panel, user_panel])

        cv2.line(combined, (panel_w, 0), (panel_w, out_h), WHITE, 2)

        cv2.putText(
            combined, f"{i + 1}/{n_frames}",
            (out_w - 110, out_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )

        writer.write(combined)

    cap_pro.release()
    cap_user.release()
    writer.release()

    # Transcode to browser-friendly H.264 so Streamlit can display it.
    with VideoFileClip(tmp_raw.name) as clip:
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio=False,
            logger=None,
        )
    try:
        os.remove(tmp_raw.name)
    except OSError:
        pass

    return output_path
