# MoveLike: compare your movement to a professional athlete.
# by Yousuf Khan

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from core.video_utils import download_youtube_video, trim_video, get_video_duration
from core.pose_extractor import extract_poses, poses_to_angle_timeseries
from core.comparison import compare_poses, score_to_label
from core.overlay_renderer import render_comparison_video
from core.insights import (
    generate_insights,
    generate_insights_no_key,
    select_joint_weights,
)


def _sweep_old_temp_files(max_age_hours: int = 24) -> None:
    # Best-effort cleanup of anything we left lying around in /tmp from
    # earlier sessions. Streamlit reruns happily leak files otherwise.
    cutoff = time.time() - max_age_hours * 3600
    for name in os.listdir(tempfile.gettempdir()):
        if not (name.startswith("tmp") and name.endswith((".mp4", ".png", ".mov"))):
            continue
        path = os.path.join(tempfile.gettempdir(), name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


_sweep_old_temp_files()

st.set_page_config(
    page_title="MoveLike",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .score-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .score-number { font-size: 3rem; font-weight: 700; }
    .score-label { font-size: 0.9rem; color: #9ca3af; margin-top: 4px; }
    .joint-row {
        display: flex;
        align-items: center;
        margin: 6px 0;
        gap: 12px;
    }
    .stProgress > div > div { border-radius: 4px; }
    h1 { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# Session state defaults
for key in ["pro_video_path", "user_video_path", "comparison_result",
            "overlay_path", "insights_text", "insights_error",
            "joint_weights_cache"]:
    if key not in st.session_state:
        st.session_state[key] = None


def save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded_file.read())
    finally:
        tmp.close()
    return tmp.name


def parse_timestamp(ts: str) -> float:
    # accepts mm:ss, hh:mm:ss, or plain seconds
    ts = ts.strip()
    if ":" in ts:
        parts = ts.split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            pass
    try:
        return float(ts)
    except ValueError:
        return 0.0


def score_colour(score: float) -> str:
    if score >= 85: return "#22c55e"
    if score >= 70: return "#84cc16"
    if score >= 55: return "#f59e0b"
    return "#ef4444"


st.markdown("# MoveLike")
st.markdown("Compare your movement technique to a professional athlete using pose analysis.")
st.caption("by Yousuf Khan")
st.divider()

# Sidebar
with st.sidebar:
    st.header("Settings")
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Optional. Get a free key at aistudio.google.com. Without it, rule-based feedback is used.",
    )
    movement_desc = st.text_input(
        "Movement description",
        value="athletic movement",
        placeholder="e.g. tennis forehand, football shot",
        help="Helps tailor the coaching tips to your sport.",
    )
    auto_onset = st.checkbox(
        "Auto-detect movement start",
        value=True,
        help="Trim each clip to where motion actually begins, so two clips "
             "that start at different times are still compared fairly.",
    )
    use_dtw = st.checkbox(
        "Warp timing to match (DTW)",
        value=True,
        help="Use dynamic time warping so fast and slow phases of the "
             "movement are paired up correctly, even if the two athletes "
             "move at different speeds. Turn off for a straight uniform "
             "resample.",
    )
    strictness = st.slider(
        "Scoring strictness",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="How harshly deviations from the pro are penalised. "
             "1.0 ≈ 25° of joint error before a joint drops below 37%. "
             "Crank it up to force tighter matching.",
    )
    st.divider()
    st.caption("**How it works**")
    st.caption("1. Load a pro video (YouTube or upload)")
    st.caption("2. Upload your own video")
    st.caption("3. Run analysis")
    st.caption("4. Get skeleton overlay + insights")


st.subheader("Step 1 — Professional Video")

pro_tab1, pro_tab2 = st.tabs(["YouTube URL", "Upload video"])

with pro_tab1:
    yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        pro_start = st.text_input("Start time", value="0:00", placeholder="mm:ss",
                                   help="Where to start the clip")
    with col2:
        pro_end = st.text_input("End time", value="0:10", placeholder="mm:ss",
                                 help="Where to end the clip")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        load_yt = st.button("Load & Trim", type="primary", use_container_width=True)

    if load_yt and yt_url:
        with st.spinner("Downloading video..."):
            try:
                tmp_dir = tempfile.mkdtemp()
                raw_path = download_youtube_video(yt_url, tmp_dir)
                duration = get_video_duration(raw_path)

                start_s = parse_timestamp(pro_start)
                end_s = parse_timestamp(pro_end)

                trimmed_path = os.path.join(tmp_dir, "pro_trimmed.mp4")
                with st.spinner(f"Trimming {start_s:.0f}s → {end_s:.0f}s..."):
                    trim_video(raw_path, start_s, end_s, trimmed_path)

                st.session_state.pro_video_path = trimmed_path
                st.success(f"Pro clip ready — {end_s - start_s:.1f}s")
            except Exception as e:
                st.error(f"Failed to load video: {e}")

with pro_tab2:
    pro_upload = st.file_uploader(
        "Upload professional video",
        type=["mp4", "mov", "avi", "mkv"],
        key="pro_upload",
    )

    if pro_upload:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            pro_up_start = st.text_input("Start time", value="0:00", placeholder="mm:ss",
                                          key="pro_up_start")
        with col2:
            pro_up_end = st.text_input("End time", value="0:10", placeholder="mm:ss",
                                        key="pro_up_end")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            trim_pro_upload = st.button("Trim", type="primary", use_container_width=True)

        if trim_pro_upload:
            with st.spinner("Processing upload..."):
                try:
                    saved = save_upload(pro_upload)
                    start_s = parse_timestamp(pro_up_start)
                    end_s = parse_timestamp(pro_up_end)
                    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    trim_video(saved, start_s, end_s, tmp_out.name)
                    st.session_state.pro_video_path = tmp_out.name
                    st.success(f"Pro clip ready — {end_s - start_s:.1f}s")
                except Exception as e:
                    st.error(f"Failed to process: {e}")

        elif pro_upload and st.session_state.pro_video_path is None:
            # Allow using the full uploaded file without trimming
            saved = save_upload(pro_upload)
            st.session_state.pro_video_path = saved

if st.session_state.pro_video_path:
    with st.expander("Preview pro clip", expanded=False):
        st.video(st.session_state.pro_video_path)

st.divider()

st.subheader("Step 2 — Your Video")

user_upload = st.file_uploader(
    "Upload your video",
    type=["mp4", "mov", "avi", "mkv"],
    key="user_upload",
    help="Record yourself doing the same movement",
)

if user_upload:
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        user_start = st.text_input("Start time", value="0:00", placeholder="mm:ss",
                                    key="user_start")
    with col2:
        user_end = st.text_input("End time", value="0:10", placeholder="mm:ss",
                                  key="user_end")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        trim_user = st.button("Trim", use_container_width=True)

    if trim_user:
        with st.spinner("Processing..."):
            try:
                saved = save_upload(user_upload)
                start_s = parse_timestamp(user_start)
                end_s = parse_timestamp(user_end)
                tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                trim_video(saved, start_s, end_s, tmp_out.name)
                st.session_state.user_video_path = tmp_out.name
                st.success(f"Your clip ready — {end_s - start_s:.1f}s")
            except Exception as e:
                st.error(f"Failed to process: {e}")
    else:
        # Use full upload without trimming
        if st.session_state.user_video_path is None:
            saved = save_upload(user_upload)
            st.session_state.user_video_path = saved

if st.session_state.user_video_path:
    with st.expander("Preview your clip", expanded=False):
        st.video(st.session_state.user_video_path)

st.divider()

st.subheader("Step 3 — Analyse")

both_ready = st.session_state.pro_video_path and st.session_state.user_video_path

if not both_ready:
    st.info("Load both videos above to run the analysis.")

run_btn = st.button(
    "Run Analysis",
    type="primary",
    disabled=not both_ready,
    use_container_width=True,
)

if run_btn and both_ready:
    progress = st.progress(0, text="Starting...")

    try:
        progress.progress(10, text="Extracting poses from professional video...")
        pro_poses, pro_fps, _ = extract_poses(st.session_state.pro_video_path)
        st.session_state.pro_fps = pro_fps

        progress.progress(35, text="Extracting poses from your video...")
        user_poses, user_fps, _ = extract_poses(st.session_state.user_video_path)
        st.session_state.user_fps = user_fps

        # Ask Gemini which joints matter most for this movement, using the
        # user's text description. Cached by description so repeat runs
        # don't re-hit the API.
        joint_weights = None
        if gemini_key and movement_desc.strip():
            cache = st.session_state.joint_weights_cache or {}
            cache_key = movement_desc.strip().lower()
            if cache_key in cache:
                joint_weights = cache[cache_key]
            else:
                progress.progress(50, text="Picking key joints for this movement...")
                try:
                    joint_weights = select_joint_weights(movement_desc, gemini_key)
                    cache[cache_key] = joint_weights
                    st.session_state.joint_weights_cache = cache
                except Exception as e:
                    st.session_state.insights_error = (
                        f"Joint-weight selection failed ({type(e).__name__}: {e}). "
                        "Falling back to equal weighting."
                    )

        progress.progress(55, text="Comparing movement patterns...")
        result = compare_poses(
            pro_poses, user_poses,
            auto_detect_onset=auto_onset,
            align_mode="dtw" if use_dtw else "uniform",
            joint_weights=joint_weights,
            strictness=strictness,
        )

        if "error" in result:
            st.error(result["error"])
            st.stop()

        st.session_state.comparison_result = result

        progress.progress(65, text="Rendering comparison video...")
        overlay_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        render_comparison_video(
            st.session_state.pro_video_path,
            st.session_state.user_video_path,
            result["aligned_pro"],
            result["aligned_user"],
            overlay_tmp.name,
        )
        st.session_state.overlay_path = overlay_tmp.name

        progress.progress(85, text="Generating coaching insights...")
        if gemini_key:
            try:
                st.session_state.insights_text = generate_insights(
                    result, gemini_key, movement_desc
                )
                st.session_state.insights_error = None
            except Exception as e:
                st.session_state.insights_error = f"{type(e).__name__}: {e}"
                st.session_state.insights_text = generate_insights_no_key(result)
        else:
            st.session_state.insights_error = None
            st.session_state.insights_text = generate_insights_no_key(result)

        progress.progress(100, text="Done!")
        time.sleep(0.5)
        progress.empty()
        st.rerun()

    except Exception as e:
        progress.empty()
        st.error(f"Analysis failed: {e}")

if st.session_state.comparison_result:
    result = st.session_state.comparison_result
    st.divider()
    st.subheader("Results")

    # If we auto-detected the movement start, let the user see where —
    # in seconds, not frames, because frame numbers are meaningless to a
    # human watching the clip.
    pro_onset = result.get("pro_onset_frame")
    user_onset = result.get("user_onset_frame")
    pro_fps = st.session_state.get("pro_fps") or 30.0
    user_fps = st.session_state.get("user_fps") or 30.0
    if (pro_onset or user_onset) and auto_onset:
        st.caption(
            f"Movement onset detected — pro clip at "
            f"{pro_onset / pro_fps:.2f}s, your clip at "
            f"{user_onset / user_fps:.2f}s. Both clips were trimmed to "
            f"that point before scoring."
        )

    low_vis = result.get("low_visibility_joints") or []
    if low_vis:
        pretty = ", ".join(j.replace("_", " ") for j in low_vis)
        st.warning(
            f"Pose detection had low confidence on these joints: **{pretty}**. "
            "Their scores are less reliable — try re-shooting with better "
            "lighting or a clearer view of the body."
        )

    pro_view = result.get("pro_view") or {}
    user_view = result.get("user_view") or {}
    merged_pairs = result.get("merged_joint_pairs") or []
    if pro_view.get("view") and user_view.get("view"):
        pro_label = pro_view["view"].replace("_", "-")
        user_label = user_view["view"].replace("_", "-")
        msg = f"Camera angle — pro: **{pro_label}**, you: **{user_label}**."
        if merged_pairs:
            pretty = ", ".join(merged_pairs)
            msg += (
                f" Left/right **{pretty}** overlap throughout both clips, "
                f"so each was merged into a single signal to avoid "
                f"double-counting."
            )
        st.caption(msg)
    if result.get("view_warning"):
        st.warning(result["view_warning"])

    jw = result.get("joint_weights") or {}
    # Joint importance classification, surfaced from the pre-analysis Gemini
    # call. Anything with normalised weight ≥ 0.10 is "major" (drives the
    # main scorecard); the rest is "minor" (collapsed under an expander).
    MAJOR_THRESHOLD = 0.10
    major_joints = [j for j, w in jw.items() if w >= MAJOR_THRESHOLD]
    minor_joints = [j for j, w in jw.items() if 0 < w < MAJOR_THRESHOLD]
    # Order both by descending weight so the most impactful joint comes first.
    major_joints.sort(key=lambda j: jw.get(j, 0.0), reverse=True)
    minor_joints.sort(key=lambda j: jw.get(j, 0.0), reverse=True)

    if jw:
        major_pretty = ", ".join(j.replace("_", " ") for j in major_joints) or "(none)"
        minor_pretty = ", ".join(j.replace("_", " ") for j in minor_joints) or "(none)"
        st.markdown(
            f"**Joint importance for this movement** — picked by Gemini "
            f"from your description (`{movement_desc}`):\n\n"
            f"- **Major:** {major_pretty}\n"
            f"- **Minor:** {minor_pretty}\n\n"
            f"The analysis below focuses on the major joints. Minor joints "
            f"are still scored but tucked under an expander."
        )

    # Overall score
    overall = result["overall_score"]
    label, colour = score_to_label(overall)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""<div class="score-card">
            <div class="score-number" style="color:{colour}">{overall:.0f}</div>
            <div style="color:{colour}; font-size:1.1rem; font-weight:600">{label}</div>
            <div class="score-label">Overall movement similarity score</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Phase scores — one row of three cards showing setup / execution /
    # follow-through. Helps a user see whether the issue is timing-localised.
    phase_scores = result.get("phase_scores") or {}
    if phase_scores:
        st.markdown("<br>", unsafe_allow_html=True)
        phase_labels = {
            "preparation": "Preparation",
            "execution": "Execution",
            "follow_through": "Follow-through",
        }
        phase_cols = st.columns(3)
        for col, (key, label) in zip(phase_cols, phase_labels.items()):
            score = phase_scores.get(key, 0.0)
            colour = score_colour(score)
            with col:
                st.markdown(
                    f"""<div class="score-card" style="padding:14px">
                    <div style="font-size:1.6rem; font-weight:700; color:{colour}">{score:.0f}</div>
                    <div class="score-label">{label}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout: video | joint scores
    vid_col, scores_col = st.columns([3, 2])

    with vid_col:
        st.markdown("**Side-by-side comparison**")
        if st.session_state.overlay_path:
            st.video(st.session_state.overlay_path)

    with scores_col:
        joint_scores = result["per_joint_scores"]

        def _render_joint_row(joint: str, score: float) -> None:
            label_text = joint.replace("_", " ").title()
            colour = score_colour(score)
            st.markdown(f"**{label_text}**")
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.progress(int(score))
            with col_b:
                st.markdown(
                    f"<span style='color:{colour}; font-weight:600'>{score:.0f}</span>",
                    unsafe_allow_html=True,
                )

        if jw:
            # Major joints take the spotlight; minors go under an expander
            # so they don't dilute the user's attention.
            st.markdown("**Joint scores — major joints for this movement**")
            for joint in major_joints:
                if joint in joint_scores:
                    _render_joint_row(joint, joint_scores[joint])

            if minor_joints:
                with st.expander(f"Show minor joints ({len(minor_joints)})"):
                    for joint in minor_joints:
                        if joint in joint_scores:
                            _render_joint_row(joint, joint_scores[joint])
        else:
            # No weights — fall back to lowest-score-first.
            st.markdown("**Joint scores**")
            for joint, score in sorted(joint_scores.items(), key=lambda x: x[1]):
                _render_joint_row(joint, score)

    st.divider()

    # Joint angle charts
    st.markdown("**Joint angle timeseries — You (orange) vs Pro (green)**")

    angle_diffs = result.get("angle_diffs", {})
    aligned_pro = result.get("aligned_pro", [])
    aligned_user = result.get("aligned_user", [])

    # Build angle timeseries from aligned poses. For any joint pair that
    # got merged ("knee" instead of "left_knee" + "right_knee"), look up
    # the kept side's raw angle from fp.joint_angles.
    overlap_keep = result.get("overlap_keep_sides") or {}
    def _raw_key(j: str) -> str:
        if "_" not in j and j in overlap_keep:
            return f"{overlap_keep[j]}_{j}"
        return j

    pro_angles = {j: [] for j in angle_diffs}
    user_angles = {j: [] for j in angle_diffs}

    for fp in aligned_pro:
        for j in angle_diffs:
            k = _raw_key(j)
            pro_angles[j].append(fp.joint_angles.get(k) if fp.joint_angles else None)

    for fp in aligned_user:
        for j in angle_diffs:
            k = _raw_key(j)
            user_angles[j].append(fp.joint_angles.get(k) if fp.joint_angles else None)

    # Charts are restricted to the major joints only — there's no value in
    # plotting elbow time-series during a squat. Within the major set, sort
    # by "importance × deficit" so the most impactful problem is on top.
    if jw and major_joints:
        ranked = sorted(
            [(j, joint_scores[j]) for j in major_joints if j in joint_scores],
            key=lambda kv: jw.get(kv[0], 0.0) * (100.0 - kv[1]),
            reverse=True,
        )
        worst_joints = [j for j, _ in ranked[:4]]
    else:
        worst_joints = [j for j, _ in sorted(joint_scores.items(), key=lambda x: x[1])[:4]]
    chart_cols = st.columns(2)

    for i, joint in enumerate(worst_joints):
        frames = list(range(len(pro_angles[joint])))
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=frames,
            y=pro_angles[joint],
            name="Pro",
            line=dict(color="#22c55e", width=2),
            connectgaps=True,
        ))
        fig.add_trace(go.Scatter(
            x=frames,
            y=user_angles[joint],
            name="You",
            line=dict(color="#f97316", width=2),
            connectgaps=True,
        ))

        fig.update_layout(
            title=joint.replace("_", " ").title(),
            xaxis_title="Frame",
            yaxis_title="Angle (°)",
            template="plotly_dark",
            height=250,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", y=1.15),
            font=dict(size=11),
        )

        with chart_cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("**Coaching Insights**")
    if st.session_state.insights_error:
        st.warning(
            f"Gemini call failed, falling back to rule-based insights.\n\n"
            f"`{st.session_state.insights_error}`"
        )
    if st.session_state.insights_text:
        st.markdown(st.session_state.insights_text)

    st.divider()
    if st.session_state.overlay_path:
        with open(st.session_state.overlay_path, "rb") as f:
            st.download_button(
                label="Download comparison video",
                data=f,
                file_name="movelike_comparison.mp4",
                mime="video/mp4",
            )
