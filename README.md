# MoveLike

**by Yousuf Khan**

MoveLike lets you drop in a clip of a professional athlete and a clip of yourself doing the same movement, and tells you how close your technique is with scores, charts, and coaching notes.

---

## What it does

1. Load a pro clip (YouTube URL + timestamps, or local file) and your own clip.
2. MediaPipe pose estimation runs on both, pulling 33 landmarks per frame.
3. Eight joint angles per frame are computed (elbows, knees, hips, shoulders).
4. Movement onset is detected and both clips are trimmed to that point.
5. DTW aligns the two sequences non-linearly, so pacing differences don't corrupt the scoring.
6. L/R joint pairs are checked individually for overlap. Side-on clips get overlapping pairs merged to avoid double-counting.
7. Each joint is scored with `exp(-|angle_diff| / tolerance_deg)`, weighted by landmark visibility. Gemini picks joint importance weights from your movement description if you have a key.
8. A per-joint angle profile (mean, min, max, ROM) is computed and passed to the coaching prompt.
9. Side-by-side skeleton overlay video is rendered using the DTW-aligned frame indices.
10. Coaching notes written by Gemini, or a rules-based fallback if no key is set.

---

## The pipeline

### Pose extraction (`core/pose_extractor.py`)

MediaPipe Tasks API with `pose_landmarker_lite`, uniformly sampled to 300 frames per clip. Frames run in `RunningMode.VIDEO` with strictly increasing timestamps. Each frame stores the full `(x, y, z, visibility)` landmark array, eight joint angles, and the original frame index.

### Normalisation (`core/comparison.py`)

Hip midpoint translated to origin, divided by torso height (hip midpoint to shoulder midpoint). Raw landmarks are meaningless to compare across athletes at different distances from the camera with different body proportions; this makes the comparison invariant to both.

### Onset detection (`core/comparison.py`)

Per-frame motion energy signal (mean landmark displacement on the normalised skeleton), smoothed with a moving average. Onset is the first frame crossing 20% of peak energy. Both sequences are sliced from their detected onset so the comparison starts at equivalent moments.

### Temporal alignment (`core/comparison.py`)

Trimming fixes when the movement starts, not differences in pace during it. A pro forehand backswing might take 200ms while yours takes 350ms; uniform resampling to the same length would pair your backswing against their backswing and their strike, which breaks the scoring.

DTW on the eight normalised joint angles per frame, with a Sakoe-Chiba band capped at 30% of sequence length. Without the band, DTW will happily match one frame against half the other video on repetitive movements. Fallback to uniform resampling available via sidebar toggle for cases where DTW finds degenerate alignments.

### Per-joint overlap detection (`core/comparison.py`)

When filmed side-on, left and right copies of a joint project onto roughly the same image point, so scoring them separately double-counts the same signal. But it's not always all-or-nothing: in three-quarter views the legs can cross while the arms don't.

Rather than a body-wide "this is a side view" decision, each L/R pair is checked individually. Median 2D distance between vertex landmarks on the normalised skeleton is measured across both clips. If it stays under ~10% of torso height in both, the pair is merged. The kept side is chosen by mean visibility across both clips. A coarser front/side/three-quarter classification still runs for the UI caption and to flag mismatched camera angles, but no longer drives merging.

### Scoring (`core/comparison.py`)

- Per-joint similarity: `exp(-|angle_diff| / tolerance_deg)`, frame-weighted by minimum landmark visibility of the defining landmark triple. Frames where a joint is occluded barely count rather than feeding garbage angles into the mean.
- Signed mean diff (user - pro) tracked alongside absolute diff so the prompt can say "under-flex" rather than just "differs by N°".
- Whole-body cosine similarity rescaled from `[baseline, 1]` to `[0, 1]`. Raw cosine on human skeletons saturates near 0.9+ because so much of the vector is shared structure; without the rescale the score never uses the low end.
- Overall score is the Gemini-weighted mean of per-joint scores, or plain cosine similarity if no weights are set.
- Strictness slider (0.5-3.0) scales angle tolerance and cosine baseline together.

### Phase scoring (`core/comparison.py`)

A single overall score is too coarse: a 72% with a clean preparation and a botched strike reads the same as 72% throughout. Aligned frames are split into thirds (preparation, execution, follow-through), each scored independently.

### Gemini joint weighting (`core/insights.py`)

Single JSON-mode call before scoring: rate each of 8 joints 0-10 for importance to this movement. Weights normalised to sum to 1 and drive the weighted score, chart selection (sorted by `weight × deficit` so the most impactful problem appears first), and prompt filtering. Joints with weight >= 0.10 are major and shown prominently; the rest are tucked under an expander. Weights are cached by movement description so repeat runs don't re-hit the API. Both calls go through a shared retry helper that backs off on 429/503s, which matters on the free tier.

### Per-joint angle stats (`core/comparison.py`)

Similarity scores are too thin to feed into proper coaching: a 70% score doesn't tell you how the user is moving, only that they're not matching. The prompt receives mean, min, max, ROM, and std per joint across aligned frames for both athletes: "pro mean 95° (range 62-168°, ROM 106°) vs user mean 110° (range 80-160°, ROM 80°)".

Body proportions (femur/tibia ratio, humerus/forearm ratio, etc.) were estimated from landmarks and fed to the prompt in an earlier version, but 2D projection error made the numbers unreliable enough that the same person filmed twice came out with significantly different proportions, so that was cut.

### Coaching notes (`core/insights.py`)

Gemini 2.5 Flash receives the filtered joint scores (major joints only, plus any minor joint with mean diff >= 25°), importance ranking, signed angle diffs, full angle profiles, and phase scores. Joints are filtered at the data layer so the model never sees irrelevant ones. Earlier versions dumped every joint into the prompt and trusted the model to ignore them; in practice it wrote a polite paragraph about each one regardless.

The prompt asks for highest-leverage corrections with current vs target angles, biomechanical reasoning, and a body cue, but leaves structure open so the model can flag cross-joint patterns ("your knee and hip are both lagging the pro by ~15° at the same point in the swing, so the issue is sequencing, not any single joint"). A strict per-joint format kills that kind of observation. Raw scores are banned from the output. Rules-based fallback uses importance-weighted deficit ordering.

### Side-by-side rendering (`core/overlay_renderer.py`)

Uses the DTW-aligned frame indices directly, so what you see matches what was scored. Frames read sequentially with `cap.grab()` / `cap.read()` rather than seeking per frame. Written as `mp4v` via OpenCV, then transcoded to H.264 via moviepy for browser compatibility.

---

## Setup

### With Docker

```bash
cd MoveLike
docker compose up --build
```

- Streamlit UI: `http://localhost:8501`
- FastAPI: `http://localhost:8000` (docs at `/docs`)

### Without Docker

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Gemini API key

Optional. Without it you get rules-based coaching and equal joint weighting. With it you get Gemini-picked weights and proper coaching notes.

---

## Sidebar controls

| Control | What it does |
|---|---|
| Gemini API Key | Unlocks joint weighting + Gemini coaching notes |
| Movement description | Used to pick joint weights ("tennis forehand", "squat", etc.) |
| Auto-detect movement start | Trims to onset before scoring |
| Warp timing to match (DTW) | Non-linear frame alignment |
| Scoring strictness | Scales angle tolerance and cosine baseline |

---

## Project structure

```
MoveLike/
├── app.py                      # Streamlit UI
├── api.py                      # FastAPI backend
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── models/                     # pose_landmarker_lite.task (auto-downloaded)
└── core/
    ├── video_utils.py          # yt-dlp download + moviepy trimming
    ├── pose_extractor.py       # MediaPipe pose extraction + joint angles
    ├── comparison.py           # normalisation, onset, DTW, scoring
    ├── overlay_renderer.py     # skeleton video renderer
    └── insights.py             # Gemini weighting + coaching + fallback
```

`core/` is UI-free. `api.py` exposes it over HTTP so any client can use the pipeline without going through Streamlit.

---

## Tips

- One rep per clip. DTW and onset detection assume a single movement.
- Side-on camera angle works best for most sports.
- 5-15 seconds is the sweet spot; longer clips are sampled down to 300 frames.
- If onset detection fires early (camera shake, etc.), turn it off and re-run.

---

## Where this is going

FastAPI backend is in place. Next is a mobile version: MediaPipe iOS Tasks SDK for on-device pose extraction, SwiftUI frontend for recording and rendering the scorecard.

Longer-term: multi-view fusion to drop the single-angle assumption, and fine-tuning the coaching prompt on expert-annotated clips.
