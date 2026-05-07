import os
import tempfile
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core.comparison import compare_poses
from core.insights import generate_insights, generate_insights_no_key, select_joint_weights
from core.overlay_renderer import render_comparison_video
from core.pose_extractor import extract_poses

app = FastAPI(title="MoveLike API")

# In-memory store mapping job_id -> overlay video path.
_videos: dict[str, str] = {}


def _serializable(obj):
    # Recursively convert numpy scalars/arrays to plain Python types so
    # FastAPI can JSON-encode the response.
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    pro_video: UploadFile = File(...),
    user_video: UploadFile = File(...),
    movement_desc: str = Form("athletic movement"),
    auto_onset: bool = Form(True),
    use_dtw: bool = Form(True),
    strictness: float = Form(1.5),
    gemini_key: str = Form(""),
):
    pro_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    pro_tmp.write(await pro_video.read())
    pro_tmp.close()

    user_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    user_tmp.write(await user_video.read())
    user_tmp.close()

    try:
        pro_poses, _, _ = extract_poses(pro_tmp.name)
        user_poses, _, _ = extract_poses(user_tmp.name)

        joint_weights = None
        if gemini_key and movement_desc.strip():
            try:
                joint_weights = select_joint_weights(movement_desc, gemini_key)
            except Exception:
                pass

        result = compare_poses(
            pro_poses,
            user_poses,
            auto_detect_onset=auto_onset,
            align_mode="dtw" if use_dtw else "uniform",
            joint_weights=joint_weights,
            strictness=strictness,
        )

        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])

        overlay_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        render_comparison_video(
            pro_tmp.name,
            user_tmp.name,
            result["aligned_pro"],
            result["aligned_user"],
            overlay_tmp.name,
        )

        if gemini_key:
            try:
                insights = generate_insights(result, gemini_key, movement_desc)
            except Exception:
                insights = generate_insights_no_key(result)
        else:
            insights = generate_insights_no_key(result)

        job_id = str(uuid.uuid4())
        _videos[job_id] = overlay_tmp.name

        response = {k: v for k, v in result.items() if k not in ("aligned_pro", "aligned_user")}
        response["insights"] = insights
        response["video_url"] = f"/video/{job_id}"

        return _serializable(response)

    finally:
        os.unlink(pro_tmp.name)
        os.unlink(user_tmp.name)


@app.get("/video/{job_id}")
def get_video(job_id: str):
    path = _videos.get(job_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")
