# Align two pose sequences in time and score how similar they are,
# frame by frame, after normalising for camera distance and athlete size.

from __future__ import annotations

import numpy as np
from core.pose_extractor import FramePose, JOINT_CONNECTIONS


def normalise_landmarks(landmarks: np.ndarray) -> np.ndarray:
    # Put the hip midpoint at the origin and divide by torso height so two
    # bodies of different sizes can be compared directly.
    lm = landmarks.copy()

    left_hip = lm[23, :3]
    right_hip = lm[24, :3]
    origin = (left_hip + right_hip) / 2
    lm[:, :3] -= origin

    left_shoulder = lm[11, :3]
    right_shoulder = lm[12, :3]
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    torso_height = np.linalg.norm(shoulder_mid - np.zeros(3)) + 1e-8
    lm[:, :3] /= torso_height

    return lm


def _motion_energy(poses: list[FramePose]) -> np.ndarray:
    # Per-frame "how much is the body moving" signal: the mean displacement
    # of all landmarks compared to the previous frame, computed on the
    # normalised skeleton so it's invariant to camera distance.
    n = len(poses)
    energy = np.zeros(n)
    prev = None
    for i, fp in enumerate(poses):
        if fp.landmarks is None:
            prev = None
            continue
        norm = normalise_landmarks(fp.landmarks)[:, :2]
        if prev is not None:
            energy[i] = float(np.mean(np.linalg.norm(norm - prev, axis=1)))
        prev = norm
    return energy


def _smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def detect_movement_start(
    poses: list[FramePose],
    threshold_ratio: float = 0.2,
    smooth_window: int = 3,
    min_idx: int = 0,
) -> int:
    # Return the index of the frame where movement begins. Uses smoothed
    # per-frame motion energy and flags the first frame whose energy crosses
    # threshold_ratio * peak. Falls back to 0 if the sequence is too flat.
    energy = _smooth(_motion_energy(poses), window=smooth_window)
    if len(energy) == 0 or energy.max() <= 1e-6:
        return 0
    threshold = energy.max() * threshold_ratio
    above = np.where(energy[min_idx:] >= threshold)[0]
    if len(above) == 0:
        return 0
    return int(min_idx + above[0])


def _pose_feature(fp: FramePose) -> np.ndarray:
    # Feature vector used for DTW: 8 joint angles divided by 180 so each
    # dimension lives in [0, 1]. Angles are already invariant to camera
    # distance and athlete size, which is what we want for matching.
    joint_names = list(JOINT_CONNECTIONS.keys())
    return np.array(
        [(fp.joint_angles.get(j) or 0.0) / 180.0 for j in joint_names],
        dtype=np.float32,
    )


def _dtw_path(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    band_ratio: float = 0.3,
) -> list[tuple[int, int]]:
    # Classic DTW with a Sakoe-Chiba band to keep alignments sane — i.e. the
    # two sequences can drift by at most band_ratio * max(len) frames apart.
    # Returns the warping path as (i, j) index pairs.
    n, m = len(feats_a), len(feats_b)
    if n == 0 or m == 0:
        return []

    band = max(1, int(band_ratio * max(n, m)))
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            d = float(np.linalg.norm(feats_a[i - 1] - feats_b[j - 1]))
            cost[i, j] = d + min(
                cost[i - 1, j],      # insert in b
                cost[i, j - 1],      # insert in a
                cost[i - 1, j - 1],  # match
            )

    # Backtrack.
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
        step = int(np.argmin(choices))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def temporal_align(
    poses_a: list[FramePose],
    poses_b: list[FramePose],
    trim_to_onset: bool = True,
    mode: str = "dtw",
) -> tuple:
    # Trim each sequence to its detected movement start, then pair frames up
    # either by uniform resampling or by DTW, which is better at matching
    # fast and slow phases to their counterparts.
    if trim_to_onset:
        start_a = detect_movement_start(poses_a)
        start_b = detect_movement_start(poses_b)
        poses_a = poses_a[start_a:]
        poses_b = poses_b[start_b:]

    valid_a = [p for p in poses_a if p.landmarks is not None]
    valid_b = [p for p in poses_b if p.landmarks is not None]

    if not valid_a or not valid_b:
        return [], []

    if mode == "dtw":
        feats_a = np.stack([_pose_feature(p) for p in valid_a])
        feats_b = np.stack([_pose_feature(p) for p in valid_b])
        path = _dtw_path(feats_a, feats_b)
        if not path:
            return [], []
        aligned_a = [valid_a[i] for i, _ in path]
        aligned_b = [valid_b[j] for _, j in path]
        return aligned_a, aligned_b

    # Uniform resample fallback.
    target_len = min(len(valid_a), len(valid_b))
    indices_a = np.linspace(0, len(valid_a) - 1, target_len).astype(int)
    indices_b = np.linspace(0, len(valid_b) - 1, target_len).astype(int)
    aligned_a = [valid_a[i] for i in indices_a]
    aligned_b = [valid_b[i] for i in indices_b]
    return aligned_a, aligned_b


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1, v2 = v1.flatten(), v2.flatten()
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    return float(np.dot(v1, v2) / denom)


def compare_poses(
    pro_poses: list[FramePose],
    user_poses: list[FramePose],
    auto_detect_onset: bool = True,
    align_mode: str = "dtw",
    joint_weights: dict[str, float] | None = None,
    strictness: float = 1.5,
) -> dict:
    # Runs the full comparison pipeline and returns the overall 0-100 score,
    # per-joint scores, per-frame similarities, and the aligned pose lists.
    # `strictness` > 1 makes small deviations cost more; < 1 is more forgiving.
    pro_onset = detect_movement_start(pro_poses) if auto_detect_onset else 0
    user_onset = detect_movement_start(user_poses) if auto_detect_onset else 0
    aligned_pro, aligned_user = temporal_align(
        pro_poses, user_poses,
        trim_to_onset=auto_detect_onset,
        mode=align_mode,
    )

    # View detection still runs to give the UI a one-line summary of the
    # camera angle and to flag obvious mismatches between the two clips.
    pro_view = detect_view_angle(pro_poses)
    user_view = detect_view_angle(user_poses)
    view_warning = None
    if pro_view["view"] != user_view["view"]:
        view_warning = (
            f"Camera angles differ (pro: {pro_view['view']}, "
            f"you: {user_view['view']}). Scoring works best when both "
            "clips are filmed from the same angle."
        )

    # The actual *grouping* decision is made per joint pair, not body-wide:
    # for each L/R pair (elbow / knee / hip / shoulder), if the two
    # landmarks project on top of each other throughout BOTH clips, we
    # treat them as one signal — zero the far side's weight pre-scoring,
    # and after scoring relabel the kept side from "left_X" to "X". This
    # works for clean side-on shots (everything overlaps), three-quarter
    # views where only the legs cross (only knee + hip overlap), and any
    # other partial-overlap geometry.
    overlap_keep = detect_overlapping_joints(pro_poses, user_poses)
    merged_joint_pairs = sorted(overlap_keep.keys())
    if joint_weights and overlap_keep:
        joint_weights = _collapse_weights_for_overlap(joint_weights, overlap_keep)

    # Tolerance parameters scale with strictness. At strictness=1 a 25°
    # angle diff scores ~37% (one "tolerance"); at strictness=2 that same
    # 25° diff scores only ~14%. The cosine baseline crops out the "shared
    # human skeleton" floor so the 0-100 range actually gets used.
    angle_tol_deg = 25.0 / max(0.1, strictness)
    cosine_baseline = min(0.98, 0.85 + (strictness - 1.0) * 0.05)

    if not aligned_pro:
        return {"error": "Could not detect poses in one or both videos."}

    joint_names = list(JOINT_CONNECTIONS.keys())
    frame_scores = []
    angle_diffs = {j: [] for j in joint_names}            # absolute, for similarity
    signed_diffs = {j: [] for j in joint_names}           # user - pro, for direction
    joint_frame_sims = {j: [] for j in joint_names}
    joint_frame_weights = {j: [] for j in joint_names}    # visibility weights

    for fp_pro, fp_user in zip(aligned_pro, aligned_user):
        norm_pro = normalise_landmarks(fp_pro.landmarks)
        norm_user = normalise_landmarks(fp_user.landmarks)

        # Whole-body similarity using just x/y (more robust than xyz).
        # Rescale from [baseline, 1] so typical values span the full 0-1
        # range rather than clustering near 1.
        raw_cos = cosine_similarity(norm_pro[:, :2], norm_user[:, :2])
        sim = max(0.0, (raw_cos - cosine_baseline) / (1.0 - cosine_baseline))
        frame_scores.append(sim)

        for j in joint_names:
            pro_angle = fp_pro.joint_angles.get(j, 0) or 0
            user_angle = fp_user.joint_angles.get(j, 0) or 0
            signed = user_angle - pro_angle
            diff = abs(signed)
            angle_diffs[j].append(diff)
            signed_diffs[j].append(signed)

            # Visibility weight: a frame contributes proportional to the
            # weakest landmark in the triple that defines this joint angle.
            # This stops occluded / off-screen joints polluting the mean
            # with junk angle values.
            a_i, b_i, c_i = JOINT_CONNECTIONS[j]
            vis_pro = min(
                float(fp_pro.landmarks[a_i, 3]),
                float(fp_pro.landmarks[b_i, 3]),
                float(fp_pro.landmarks[c_i, 3]),
            )
            vis_user = min(
                float(fp_user.landmarks[a_i, 3]),
                float(fp_user.landmarks[b_i, 3]),
                float(fp_user.landmarks[c_i, 3]),
            )
            w = min(vis_pro, vis_user)
            joint_frame_weights[j].append(w)

            # Exponential decay: a diff equal to `angle_tol_deg` scores ~37%,
            # double that scores ~14%. Much sharper than the old linear curve.
            joint_sim = float(np.exp(-diff / angle_tol_deg))
            joint_frame_sims[j].append(joint_sim)

    # Visibility-weighted mean per joint. If a joint had zero visibility
    # across the whole clip we fall back to the unweighted mean so the score
    # doesn't go NaN — but flag it so the UI can warn.
    per_joint_scores: dict[str, float] = {}
    low_visibility_joints: list[str] = []
    for j in joint_names:
        sims = np.array(joint_frame_sims[j], dtype=float)
        weights = np.array(joint_frame_weights[j], dtype=float)
        if weights.sum() < 1e-3:
            per_joint_scores[j] = float(sims.mean()) * 100 if len(sims) else 0.0
            low_visibility_joints.append(j)
        else:
            per_joint_scores[j] = float(np.average(sims, weights=weights)) * 100
            if weights.mean() < 0.5:
                low_visibility_joints.append(j)

    if joint_weights:
        # Weighted mean of per-joint scores, emphasising the joints Gemini
        # picked as most relevant to this specific movement.
        overall_score = sum(
            per_joint_scores[j] * joint_weights.get(j, 0.0)
            for j in joint_names
        )
    else:
        overall_score = float(np.clip(np.mean(frame_scores), 0.0, 1.0)) * 100

    pro_angle_stats = compute_angle_stats(aligned_pro, joint_names)
    user_angle_stats = compute_angle_stats(aligned_user, joint_names)

    # Phase scoring — split the aligned sequence into thirds (preparation,
    # execution, follow-through). Coaches care which phase is bad, not just
    # the overall number. Boundaries are by frame index, not by motion energy
    # because motion peaks aren't reliable across all sports.
    phase_scores = _compute_phase_scores(frame_scores)

    # Signed mean diff per joint so the prompt can tell the user whether
    # they're over- or under-rotating, not just "off by 12°".
    signed_mean_diffs: dict[str, float] = {}
    for j in joint_names:
        weights = np.array(joint_frame_weights[j], dtype=float)
        signed = np.array(signed_diffs[j], dtype=float)
        if weights.sum() < 1e-3 or len(signed) == 0:
            continue
        signed_mean_diffs[j] = round(float(np.average(signed, weights=weights)), 1)

    # For each joint pair we decided to merge, drop the far-side label and
    # rename the kept side to the un-sided base name ("left_knee" → "knee").
    # Pairs that *don't* overlap stay untouched, so a three-quarter clip
    # where only the legs overlap still reports left vs right elbow.
    if overlap_keep:
        per_joint_scores      = _merge_pairs(per_joint_scores,      overlap_keep)
        angle_diffs           = _merge_pairs(angle_diffs,           overlap_keep)
        signed_mean_diffs     = _merge_pairs(signed_mean_diffs,     overlap_keep)
        joint_weights         = _merge_pairs(joint_weights,         overlap_keep, renormalise=True)
        pro_angle_stats       = _merge_pairs(pro_angle_stats,       overlap_keep)
        user_angle_stats      = _merge_pairs(user_angle_stats,      overlap_keep)
        low_visibility_joints = _merge_pairs_list(low_visibility_joints, overlap_keep)

    return {
        "overall_score": round(overall_score, 1),
        "per_joint_scores": per_joint_scores,
        "frame_scores": frame_scores,
        "angle_diffs": angle_diffs,
        "aligned_pro": aligned_pro,
        "aligned_user": aligned_user,
        "n_frames_compared": len(aligned_pro),
        "pro_onset_frame": pro_onset,
        "user_onset_frame": user_onset,
        "joint_weights": joint_weights,
        "pro_angle_stats": pro_angle_stats,
        "user_angle_stats": user_angle_stats,
        "pro_view": pro_view,
        "user_view": user_view,
        "merged_joint_pairs": merged_joint_pairs,
        "overlap_keep_sides": overlap_keep,
        "view_warning": view_warning,
        "phase_scores": phase_scores,
        "signed_mean_diffs": signed_mean_diffs,
        "low_visibility_joints": low_visibility_joints,
    }


# Landmark index groups used for view-angle detection.
_LEFT_BODY_IDS  = [11, 13, 15, 23, 25, 27]   # shoulder, elbow, wrist, hip, knee, ankle
_RIGHT_BODY_IDS = [12, 14, 16, 24, 26, 28]


def detect_view_angle(poses: list[FramePose]) -> dict:
    # Classify the camera angle as front / side / three_quarter based on how
    # wide the shoulders and hips project in the image. In a true side-on
    # shot the L and R hip landmarks almost overlap, so their horizontal
    # distance (after normalising by torso length) drops near zero.
    # Also identifies which side of the body is facing the camera by
    # comparing average landmark visibility left vs right.
    valid = [p for p in poses if p.landmarks is not None]
    if not valid:
        return {"view": "unknown", "near_side": None,
                "shoulder_width": None, "hip_width": None}

    shoulder_widths: list[float] = []
    hip_widths: list[float] = []
    left_vis: list[float] = []
    right_vis: list[float] = []

    for fp in valid:
        lm = normalise_landmarks(fp.landmarks)
        shoulder_widths.append(abs(float(lm[11, 0] - lm[12, 0])))
        hip_widths.append(abs(float(lm[23, 0] - lm[24, 0])))
        left_vis.append(float(np.mean([fp.landmarks[i, 3] for i in _LEFT_BODY_IDS])))
        right_vis.append(float(np.mean([fp.landmarks[i, 3] for i in _RIGHT_BODY_IDS])))

    sw_med = float(np.median(shoulder_widths))
    hw_med = float(np.median(hip_widths))

    if sw_med < 0.18 and hw_med < 0.12:
        view = "side"
    elif sw_med > 0.30 or hw_med > 0.18:
        view = "front"
    else:
        view = "three_quarter"

    near = None
    lv, rv = float(np.mean(left_vis)), float(np.mean(right_vis))
    if view in ("side", "three_quarter") and abs(lv - rv) > 0.03:
        near = "left" if lv > rv else "right"

    return {
        "view": view,
        "near_side": near,
        "shoulder_width": round(sw_med, 3),
        "hip_width": round(hw_med, 3),
    }


PHASE_NAMES = ("preparation", "execution", "follow_through")


def _compute_phase_scores(frame_scores: list[float]) -> dict[str, float]:
    # Split the aligned frame-similarity scores into thirds and average each.
    # Reported as 0-100 so they line up with the overall score scale.
    if not frame_scores:
        return {p: 0.0 for p in PHASE_NAMES}
    n = len(frame_scores)
    cut1 = max(1, n // 3)
    cut2 = max(cut1 + 1, (2 * n) // 3)
    chunks = [
        frame_scores[:cut1],
        frame_scores[cut1:cut2],
        frame_scores[cut2:],
    ]
    return {
        name: round(float(np.mean(chunk)) * 100, 1) if chunk else 0.0
        for name, chunk in zip(PHASE_NAMES, chunks)
    }


# Joint pairs we'll consider merging when their L/R copies project onto
# each other throughout the clip. Indices are the *vertex* landmark of
# each joint angle (the middle of the three points).
_JOINT_PAIR_VERTICES = {
    "elbow":    (13, 14),
    "knee":     (25, 26),
    "hip":      (23, 24),
    "shoulder": (11, 12),
}


def _per_pair_metrics(
    poses: list[FramePose],
    l_idx: int,
    r_idx: int,
) -> tuple[list[float], list[float], list[float]]:
    # Per-frame normalised distance between the two vertex landmarks, plus
    # mean visibility on each side.
    distances: list[float] = []
    l_vis: list[float] = []
    r_vis: list[float] = []
    for fp in poses:
        if fp.landmarks is None:
            continue
        lm = normalise_landmarks(fp.landmarks)
        distances.append(float(np.linalg.norm(lm[l_idx, :2] - lm[r_idx, :2])))
        l_vis.append(float(fp.landmarks[l_idx, 3]))
        r_vis.append(float(fp.landmarks[r_idx, 3]))
    return distances, l_vis, r_vis


def detect_overlapping_joints(
    pro_poses: list[FramePose],
    user_poses: list[FramePose],
    distance_threshold: float = 0.10,
) -> dict[str, str]:
    # For each L/R joint pair, decide whether the two landmarks project
    # close enough to each other throughout BOTH clips that scoring them
    # separately just double-counts the same signal. Returns a map of
    # {base_joint: keep_side} for every pair we want to merge; pairs that
    # genuinely move independently are absent. Distance is on the
    # hip-centred / torso-scaled skeleton, so 0.10 is "10% of torso height".
    out: dict[str, str] = {}
    for base, (l_idx, r_idx) in _JOINT_PAIR_VERTICES.items():
        pro_d, pro_lv, pro_rv = _per_pair_metrics(pro_poses, l_idx, r_idx)
        usr_d, usr_lv, usr_rv = _per_pair_metrics(user_poses, l_idx, r_idx)
        if not pro_d or not usr_d:
            continue
        if (
            float(np.median(pro_d)) < distance_threshold
            and float(np.median(usr_d)) < distance_threshold
        ):
            # Pick the side with higher combined visibility — that's the
            # one closer to the camera and therefore the cleaner signal.
            l_score = float(np.mean(pro_lv + usr_lv))
            r_score = float(np.mean(pro_rv + usr_rv))
            out[base] = "left" if l_score >= r_score else "right"
    return out


def _collapse_weights_for_overlap(
    weights: dict[str, float],
    overlap_keep: dict[str, str],
) -> dict[str, float]:
    # For each merged pair, zero out the discarded side's weight, then
    # renormalise the whole dict so weights still sum to 1.
    if not weights or not overlap_keep:
        return weights
    adjusted = dict(weights)
    for base, keep in overlap_keep.items():
        far = "right" if keep == "left" else "left"
        far_key = f"{far}_{base}"
        if far_key in adjusted:
            adjusted[far_key] = 0.0
    total = sum(adjusted.values()) or 1.0
    return {j: w / total for j, w in adjusted.items()}


def _merge_pairs(
    data: dict,
    overlap_keep: dict[str, str],
    renormalise: bool = False,
) -> dict:
    # Collapse per-pair: for each base joint in overlap_keep, the kept side
    # is renamed to the un-sided base name and the other side is dropped.
    # Joints not listed in overlap_keep are left alone.
    if not data or not overlap_keep:
        return data
    rename: dict[str, str] = {}    # original key → new key
    drop: set[str] = set()
    for base, keep in overlap_keep.items():
        far = "right" if keep == "left" else "left"
        rename[f"{keep}_{base}"] = base
        drop.add(f"{far}_{base}")
    out: dict = {}
    for k, v in data.items():
        if k in drop:
            continue
        out[rename.get(k, k)] = v
    if renormalise:
        total = sum(v for v in out.values() if isinstance(v, (int, float))) or 1.0
        out = {k: (v / total if isinstance(v, (int, float)) else v) for k, v in out.items()}
    return out


def _merge_pairs_list(items: list[str], overlap_keep: dict[str, str]) -> list[str]:
    if not items or not overlap_keep:
        return items
    rename: dict[str, str] = {}
    drop: set[str] = set()
    for base, keep in overlap_keep.items():
        far = "right" if keep == "left" else "left"
        rename[f"{keep}_{base}"] = base
        drop.add(f"{far}_{base}")
    return [rename.get(j, j) for j in items if j not in drop]


# Landmark pairs used when estimating body proportions from 2D poses.
# All lengths come out expressed as multiples of torso height because
# normalise_landmarks() divides through by that.
_SEGMENT_PAIRS = {
    "femur_L":    (23, 25),   # left hip → left knee
    "femur_R":    (24, 26),
    "tibia_L":    (25, 27),   # left knee → left ankle
    "tibia_R":    (26, 28),
    "humerus_L":  (11, 13),   # left shoulder → left elbow
    "humerus_R":  (12, 14),
    "forearm_L":  (13, 15),   # left elbow → left wrist
    "forearm_R":  (14, 16),
    "shoulder_w": (11, 12),
    "hip_w":      (23, 24),
}


def estimate_body_proportions(poses: list[FramePose], min_visibility: float = 0.5) -> dict:
    # Rough anthropometric measurements from 2D pose. These are noisy
    # because a femur projected at 45° from the camera reads shorter than
    # one that's perpendicular, so I take the median across frames for
    # each segment to reject the bad views. Numbers come out as multiples
    # of torso height.
    valid = [p for p in poses if p.landmarks is not None]
    if not valid:
        return {}

    collected: dict[str, list[float]] = {k: [] for k in _SEGMENT_PAIRS}
    for fp in valid:
        lm = normalise_landmarks(fp.landmarks)
        for name, (a, b) in _SEGMENT_PAIRS.items():
            if fp.landmarks[a, 3] < min_visibility or fp.landmarks[b, 3] < min_visibility:
                continue
            dist = float(np.linalg.norm(lm[a, :2] - lm[b, :2]))
            collected[name].append(dist)

    raw = {
        name: round(float(np.median(vs)), 3)
        for name, vs in collected.items() if vs
    }

    def avg_lr(base: str) -> float | None:
        l, r = raw.get(f"{base}_L"), raw.get(f"{base}_R")
        if l is not None and r is not None:
            return round((l + r) / 2, 3)
        return l if l is not None else r

    out: dict[str, float] = {}
    for base in ("femur", "tibia", "humerus", "forearm"):
        v = avg_lr(base)
        if v is not None:
            out[base] = v
    if "shoulder_w" in raw:
        out["shoulder_width"] = raw["shoulder_w"]
    if "hip_w" in raw:
        out["hip_width"] = raw["hip_w"]

    # Compound ratios that actually matter for coaching — long femur vs
    # short tibia changes squat depth mechanics, humerus vs forearm
    # changes bench/pull-up leverage, etc.
    if "femur" in out and "tibia" in out and out["tibia"]:
        out["femur_to_tibia"] = round(out["femur"] / out["tibia"], 2)
    if "humerus" in out and "forearm" in out and out["forearm"]:
        out["humerus_to_forearm"] = round(out["humerus"] / out["forearm"], 2)
    if "shoulder_width" in out and "hip_width" in out and out["hip_width"]:
        out["shoulder_to_hip"] = round(out["shoulder_width"] / out["hip_width"], 2)

    return out


def compute_angle_stats(aligned_poses: list[FramePose], joint_names: list[str]) -> dict:
    # Per-joint descriptive stats across the aligned frames: mean, min,
    # max, range of motion, and std. Much richer than a single avg diff.
    stats: dict[str, dict] = {}
    for j in joint_names:
        vals = [
            fp.joint_angles.get(j)
            for fp in aligned_poses
            if fp.joint_angles and fp.joint_angles.get(j) is not None
        ]
        if not vals:
            stats[j] = None
            continue
        arr = np.array(vals, dtype=float)
        stats[j] = {
            "mean_deg":  round(float(arr.mean()), 1),
            "min_deg":   round(float(arr.min()), 1),
            "max_deg":   round(float(arr.max()), 1),
            "range_deg": round(float(arr.max() - arr.min()), 1),
            "std_deg":   round(float(arr.std()), 1),
        }
    return stats


def score_to_label(score: float) -> tuple[str, str]:
    # Map a 0-100 score to a human label and a hex colour.
    if score >= 85:
        return "Excellent", "#22c55e"
    elif score >= 70:
        return "Good", "#84cc16"
    elif score >= 55:
        return "Fair", "#f59e0b"
    else:
        return "Needs Work", "#ef4444"
