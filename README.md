# MoveLike

**by Yousuf Khan**

MoveLike lets you drop in a clip of a professional athlete and a clip of yourself doing the same movement, and it tells you, with numbers, charts, and coaching notes, how close your technique is to theirs.

I built it because watching yourself on video and *thinking* you look like Djokovic isn't the same as a computer measuring the angle of your shoulder rotation frame by frame and telling you it's 23° shy. The point of the tool is to take the guesswork out.

---

## What it actually does, end to end

1. You load a pro clip (YouTube URL + timestamps, or a local file) and your own clip.
2. The app runs MediaPipe pose estimation on every sampled frame of both videos and pulls out 33 body landmarks per frame.
3. From those landmarks it computes eight joint angles per frame (both elbows, knees, hips, and shoulders).
4. It detects when the actual *movement* starts in each clip (the two clips almost never start at exactly the same moment) and trims both to that point.
5. It uses Dynamic Time Warping to pair up frames across the two clips, so your slow backswing gets matched against the pro's slow backswing even if your overall pace is different.
6. It detects the camera angle on each clip. If both clips are side-on with the same side facing the camera, it folds the far-side left/right joints into the near-side ones. From a side view they overlap and measuring them separately just double-counts the same signal.
7. It scores each joint using an exponential decay curve, weights the joints by how much they matter for the specific movement (Gemini picks the weights if you've given it a key), and produces an overall 0-100 score plus per-joint scores.
8. It computes a full per-joint angle profile (mean, min, max, range of motion) so the coaching prompt can reason about specific degrees rather than just similarity percentages.
9. It renders a side-by-side skeleton overlay video using the same warped alignment, so you can actually see where you and the pro diverge.
10. It writes coaching notes, Gemini-powered if you've got a key set, or a simple rule-based fallback otherwise. The prompt only sees the joints that actually matter for this movement, and gives the model leeway to flag cross-joint patterns rather than mechanically going through every joint.

Everything is tunable from the sidebar: strictness, DTW on/off, onset trim on/off, movement description, API key.

---

## The pipeline, in detail

### Pose extraction (`core/pose_extractor.py`)

I use MediaPipe's Tasks API with the `pose_landmarker_lite` model (~6 MB, auto-downloaded on first run). To keep long clips manageable, I uniformly sample up to 300 frames from each video. Any more and extraction gets slow without adding useful detail. Each sampled frame gets run through the landmarker in `RunningMode.VIDEO`, which needs strictly increasing timestamps.

For each frame I store:
- The full 33-landmark array `(x, y, z, visibility)`
- Eight joint angles computed from triples of landmarks (`shoulder → elbow → wrist` gives the elbow angle, etc.)
- The original frame index, so later stages can map back to the source video

### Normalisation (`core/comparison.py`)

Two athletes don't stand at the same distance from the camera and don't have the same body proportions, so raw landmarks are meaningless to compare directly. Before any similarity math I translate each skeleton so the midpoint of the hips is at the origin, then divide everything by torso height (hip midpoint → shoulder midpoint). That makes the comparison invariant to camera distance and body size.

### Onset detection (`core/comparison.py`)

Even if both clips are trimmed to roughly the same section, the athlete in one might start moving half a second later than in the other. I compute a per-frame "motion energy" signal (the average landmark displacement from the previous frame, on the normalised skeleton), smooth it with a small moving average, then find the first frame where energy crosses 20% of the peak. That's the movement onset.

Both sequences get sliced from their detected onset, so the comparison starts from equivalent moments in each. The detected frame indices are surfaced in the UI so you can sanity-check the detection.

### Temporal alignment (`core/comparison.py`)

Trimming fixes *when* the movement starts; it doesn't fix differences in *pace* during the movement. A tennis pro's forehand backswing might take 200 ms while mine takes 350. Uniform resampling to the same length would pair my backswing against their backswing *and* their strike, which ruins the scoring.

I solve this with Dynamic Time Warping on a per-frame feature vector of the eight normalised joint angles. DTW finds the optimal non-linear mapping between the two sequences. I apply a **Sakoe-Chiba band** (max 30% of sequence length) to keep the warping physically reasonable. Without the band DTW will happily match one frame against half the other video.

There's a sidebar toggle to fall back to uniform resampling if DTW produces weird results on a specific clip (e.g. very repetitive movements where DTW can find degenerate alignments).

### Per-joint overlap detection (`core/comparison.py`)

When a clip is filmed side-on, the left and right copies of each joint project onto roughly the same point in the image, so scoring "left knee" and "right knee" separately just double-counts the same signal. But that's not always all-or-nothing: in three-quarter views the legs can cross while the arms don't, in some yoga poses one limb sits behind the other only at certain phases, and so on. Rather than make a body-wide "this is a side view" decision, I check each L/R joint pair individually.

For each pair (elbow, knee, hip, shoulder) I measure the median 2D distance between the two vertex landmarks on the hip-centred / torso-scaled skeleton, across both clips. If that distance stays under ~10% of torso height in *both* the pro clip and the user clip, the pair gets merged: the discarded side's weight is zeroed before scoring, then post-scoring the kept side is renamed (`left_knee` → `knee`) across the per-joint scores, angle diffs, weights, and stats. Joints that don't overlap stay as `left_X` / `right_X` and get scored independently.

The "kept" side is chosen per pair using mean visibility across both clips, so the pair always reports from whichever side the camera actually sees.

A coarser front / side / three_quarter view classification still runs for the UI caption and to warn the user when the two clips were obviously filmed from different angles, but it no longer drives the merge decision.

### Scoring (`core/comparison.py`)

For each aligned frame pair:

- **Per-joint similarity**: `exp(-|angle_diff| / tolerance_deg)`, then averaged across the clip with each frame's contribution weighted by the *minimum landmark visibility* of the three points that define that joint angle. Frames where a joint is occluded or off-screen barely count, instead of feeding garbage angles into the mean. Joints whose mean visibility falls below 0.5 across the whole clip get flagged in the UI so the user knows the score on that joint isn't trustworthy.
- **Signed angle diff**: alongside the absolute diff used for similarity, I also track the *signed* mean diff (user − pro) per joint. Without this the prompt can't tell over-rotation from under-rotation; with it, the coaching notes can say "you under-flex your knee by ~12°" instead of "your knee differs by 12°".
- **Whole-body similarity**: cosine similarity of the full normalised landmark set, rescaled from `[baseline, 1]` to `[0, 1]`. Raw cosine similarity is almost always 0.9+ on human skeletons because so much of the vector is shared structure, so without the rescale the score never really uses the low end.

The overall score is either:
- A **weighted mean** of per-joint scores if Gemini has picked joint importance weights, or
- The plain cosine similarity otherwise

A sidebar **strictness** slider (0.5 – 3.0) scales the angle tolerance and the cosine baseline together, so you can tighten or loosen the whole system with one knob.

### Phase scoring (`core/comparison.py`)

A single overall score is too coarse for coaching: a 72% with a clean preparation and a botched strike reads the same as 72% throughout. After scoring, the aligned frames get split into thirds (preparation, execution, follow-through) and each third gets its own 0-100 score, surfaced as three cards in the UI. If your overall is 70 but execution is sitting at 50, you know exactly where to focus.

### Gemini joint weighting (`core/insights.py`)

When you set a Gemini key and a movement description, the app makes a single JSON-mode call **before scoring**: "for this movement, rate each of these 8 joints 0-10 for how critical it is". The integers are normalised to weights summing to 1 and then drive everything downstream:

- **Major / minor split.** Joints with weight ≥ 0.10 are classified as *major* and shown prominently in the joint-scores panel. Minor joints are still scored but tucked under an expander so they don't dilute the eye. There's no point making a user read about elbow technique during a squat.
- **Weighted overall score** - the per-joint similarities get averaged with these weights instead of equally.
- **Chart selection** - only major joints get angle timeseries charts, sorted within that set by `weight × deficit` so the most *impactful* problem appears first.
- **Coaching-prompt filtering**: the joints sent to Gemini for the coaching notes are also restricted to majors (plus any minor joint with an extreme deviation), so the response never pads with irrelevant feedback.

The app shows the major/minor classification at the top of the results so the user can see *why* certain joints are being deprioritised, e.g. for "tennis forehand" Gemini will flag shoulder + elbow + hip as major and push knees into the minor pile.

Weights are cached on disk under `.cache/joint_weights.json` keyed by the movement description, so typing "tennis forehand" doesn't re-hit the API on subsequent runs even across full app restarts. Both the weight-selection call and the coaching-notes call run through a shared retry helper that backs off on transient 429 / 503 errors. This matters on the Gemini free tier, which throttles aggressively.

### Per-joint angle stats (`core/comparison.py`)

Similarity scores alone are too thin to feed into proper coaching: a 70% score doesn't tell you *how* the user is moving, only that they're not quite matching. So before I hand data to the coaching prompt I compute a per-joint angle profile for each athlete: mean, min, max, range of motion, and standard deviation across the aligned frames. Rather than "your knee scored 62", the prompt sees "pro mean 95° (range 62–168°, ROM 106°) vs user mean 110° (range 80–160°, ROM 80°)".

(I previously also estimated body proportions from the landmarks (femur/tibia ratio, humerus/forearm ratio, etc.) and fed them to the coaching prompt, but 2D projection error made the numbers unreliable enough that the same person filmed twice came out with significantly different proportions, so I pulled it out.)

### Coaching notes (`core/insights.py`)

After scoring, I send Gemini (default `gemini-2.5-flash`) a prompt built from:

- The per-joint similarity scores, **filtered** to the joints that actually matter for this movement (weight ≥ 0.10), plus any minor joint whose mean diff is genuinely extreme (≥ 25°).
- The joint importance ranking, in order.
- The mean angle diff per joint (absolute and signed, so the model can phrase corrections directionally rather than just "off by N°").
- The full pro-vs-user angle profile (mean / min / max / ROM) for the same filtered joint set.
- The three phase scores (preparation / execution / follow-through) so the model can localise problems to the part of the movement they actually live in.

The filtering is the important bit. Earlier versions dumped every joint into the prompt and trusted the model to ignore the irrelevant ones. In practice it would write a polite paragraph about each one anyway, padding the output with feedback on joints that don't matter for the movement. Cutting them at the data layer means the model never sees them.

The prompt asks for a focused response with these minimum points:
1. A short summary of technique quality.
2. The handful of highest-leverage corrections: current vs target angle in degrees, the biomechanical reason the change matters, and a concrete body cue the user can feel.
3. One thing the user is doing well.
4. A drill or focus for the next session.

But it deliberately doesn't lock the model into rigid section headings. Instead it tells the model to use its own judgement about what the data is saying and to flag cross-joint patterns when they show up, e.g. "your knee and hip are both lagging the pro by ~15° at the same point in the swing, so the issue is sequencing, not any single joint". That sort of observation is exactly what an LLM is good at and what a strict per-joint format kills.

The prompt also bans raw scores from the output. Gemini has to translate numbers into coaching language.

If there's no API key, or the call fails, a rules-based fallback picks "areas to improve" using importance-weighted deficit. The priority ordering stays consistent with the Gemini path.

### Side-by-side rendering (`core/overlay_renderer.py`)

The overlay video uses the frame indices of the aligned poses directly, so what you see matches what was actually scored, onset-trimmed and DTW-warped. Frames are read sequentially (`cap.grab()` to skip, then `cap.read()` on the ones we need) rather than seeking per frame, which is much faster in practice.

Each panel gets bones and joint dots drawn in its team colour (pro: green, you: orange), plus a bold label and frame counter. The final render is written as `mp4v` via OpenCV, then transcoded to H.264 via moviepy so the browser's HTML5 video element can actually play it inline. Without the transcode step Streamlit just shows a greyed-out player.

---

## Setup

```bash
cd MoveLike
python3 -m venv venv
source venv/bin/activate       # mac/linux
# or: venv\Scripts\activate    # windows
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. First run will download the MediaPipe pose model (~6 MB) the first time you hit "Run Analysis".

### Optional: Gemini API key

Not required (the app works end to end without it), but with a key you get:
- Gemini-picked joint importance weights (affects scoring, UI ordering, and which charts show)
- Real coaching notes in plain English rather than the rules-based fallback

---

## Sidebar controls

| Control | What it does |
|---|---|
| Gemini API Key | Unlocks joint weighting + AI-written coaching notes |
| Movement description | Text hint used to pick joint weights ("tennis forehand", "squat", etc.) |
| Auto-detect movement start | Trims each clip to where motion actually begins |
| Warp timing to match (DTW) | Pairs up frames non-linearly so fast/slow phases align |
| Scoring strictness | Tightens or loosens the whole scoring system via a single slider |

---

## Project structure

```
MoveLike/
├── app.py                      # streamlit UI, wires everything together
├── setup.sh                    # one-shot venv + install helper
├── requirements.txt
├── models/                     # pose_landmarker_lite.task goes here (auto-downloaded)
└── core/
    ├── video_utils.py          # yt-dlp download + moviepy trimming
    ├── pose_extractor.py       # MediaPipe Tasks pose extraction + joint angles
    ├── comparison.py           # normalisation, onset detection, DTW, scoring
    ├── overlay_renderer.py     # side-by-side skeleton video renderer
    └── insights.py             # Gemini weight selection + coaching notes + rules fallback
```

The `core/` modules are deliberately UI-free: they take paths and return data. That means I can lift them into a FastAPI service later for a mobile frontend without rewriting any of the analysis logic.

---

## Tips for better results

- **One repetition per clip.** One tennis swing, one squat rep, one golf shot. DTW and onset detection both assume a single movement.
- **Same camera angle.** Side-on works best for most sports. The pose estimator is 2D-ish; you need the camera to see the joints that matter.
- **Decent lighting.** MediaPipe falls over on dark or low-contrast footage. If the skeleton flickers in the preview, lighting's the culprit.
- **5–15 seconds** is the sweet spot. Longer clips still work but get sampled down to 300 frames max.
- **If the onset detection looks wrong** (e.g. camera shake triggered an early onset), turn it off in the sidebar and re-run.

---

## Where this is going

The analysis pipeline is UI-agnostic, so the obvious next step is a mobile version:
1. Wrap `core/` in a FastAPI backend
2. Use MediaPipe's iOS Tasks SDK for on-device pose estimation
3. Build a SwiftUI frontend that records/uploads clips and renders the scorecard

Longer-term I'd like to replace the single-angle camera assumption with multi-view fusion, and experiment with fine-tuning the Gemini prompt on a small dataset of expert-annotated clips so the coaching notes improve over time.
