# Turns comparison numbers into plain-English coaching notes, either by
# asking Gemini (when a key is provided) or via a simple rules fallback.

from __future__ import annotations

import os
import json
import time
import hashlib
import numpy as np
from google import genai

from core.pose_extractor import JOINT_CONNECTIONS


_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
_WEIGHTS_CACHE = os.path.join(_CACHE_DIR, "joint_weights.json")


def _load_weights_cache() -> dict:
    if not os.path.exists(_WEIGHTS_CACHE):
        return {}
    try:
        with open(_WEIGHTS_CACHE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_weights_cache(cache: dict) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_WEIGHTS_CACHE, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass


def _generate_with_retry(client, *, model, contents, config=None, max_retries: int = 3):
    # Gemini's free tier 429s readily; retry with exponential backoff so a
    # transient quota blip doesn't dump the user back to the rules fallback.
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            kwargs = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            return client.models.generate_content(**kwargs)
        except Exception as e:  # google-genai uses ClientError for 4xx/5xx
            last_err = e
            msg = str(e).lower()
            transient = any(s in msg for s in ("429", "503", "deadline", "unavailable"))
            if not transient or attempt == max_retries - 1:
                raise
            time.sleep(0.8 * (2 ** attempt))
    if last_err:
        raise last_err


def _parse_weight_response(raw_json: str, joints: list[str]) -> dict[str, float]:
    raw = json.loads(raw_json)
    scores = {j: max(0.0, float(raw.get(j, 1))) for j in joints}
    total = sum(scores.values()) or float(len(joints))
    return {j: s / total for j, s in scores.items()}


def select_joint_weights(
    movement_description: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> dict[str, float]:
    # Ask Gemini which joints matter most for this movement. Returns a dict
    # {joint: weight} where weights are in [0, 1] and sum to 1, so they can
    # drop straight into a weighted mean.
    joints = list(JOINT_CONNECTIONS.keys())

    prompt = (
        "You are a sports biomechanics coach. For the movement below, rate "
        "how important each listed joint is when judging technique, on a "
        "0-10 integer scale (10 = critical, 0 = irrelevant).\n\n"
        f"Movement: {movement_description}\n\n"
        f"Joints: {joints}\n\n"
        "Respond with JSON only. Keys must be the exact joint names above. "
        "Every joint must be included. Example:\n"
        '{"left_elbow": 9, "right_elbow": 9, "left_knee": 2, ...}'
    )

    # Disk cache so the same description doesn't re-hit Gemini on every
    # session — important on the free tier where quota resets daily.
    cache_key = hashlib.sha1(
        f"{model}:{movement_description.strip().lower()}".encode()
    ).hexdigest()
    cache = _load_weights_cache()
    if cache_key in cache:
        return cache[cache_key]

    client = genai.Client(api_key=api_key)
    response = _generate_with_retry(
        client,
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    weights = _parse_weight_response(response.text, joints)
    cache[cache_key] = weights
    _save_weights_cache(cache)
    return weights


def generate_insights(
    comparison_result: dict,
    api_key: str,
    movement_description: str = "the movement",
) -> str:
    # Ask Gemini to turn the joint data into coaching notes (markdown),
    # with enough detail that it can reason biomechanically about *why*
    # the user's mechanics differ and how their anthropometry should
    # shape the correction.
    client = genai.Client(api_key=api_key)

    per_joint         = comparison_result.get("per_joint_scores", {})
    angle_diffs       = comparison_result.get("angle_diffs", {})
    overall           = comparison_result.get("overall_score", 0)
    joint_weights     = comparison_result.get("joint_weights") or {}
    pro_angle_stats   = comparison_result.get("pro_angle_stats", {})
    user_angle_stats  = comparison_result.get("user_angle_stats", {})
    merged_pairs      = comparison_result.get("merged_joint_pairs") or []
    signed_diffs      = comparison_result.get("signed_mean_diffs", {})
    phase_scores      = comparison_result.get("phase_scores", {})

    # Collapse per-frame angle diffs down to one number per joint.
    avg_diffs = {
        joint: round(float(np.mean([d for d in diffs if d is not None])), 1)
        for joint, diffs in angle_diffs.items()
        if diffs
    }

    # Rank joints by how much they matter for this movement and only feed
    # the prompt the joints worth talking about. Minor joints are dropped
    # unless their angle diff is genuinely extreme, so the model isn't
    # tempted to pad the response with irrelevant observations.
    EXTREME_DIFF_DEG = 25.0
    if joint_weights:
        ranked = sorted(joint_weights.items(), key=lambda kv: kv[1], reverse=True)
        n_keep = max(3, min(5, sum(1 for _, w in ranked if w >= 0.10)))
        key_joints = [j for j, _ in ranked[:n_keep]]
        extreme_minors = [
            j for j, w in ranked[n_keep:]
            if avg_diffs.get(j, 0) >= EXTREME_DIFF_DEG
        ]
        relevant_joints = key_joints + extreme_minors
        priority_block = (
            f"\n**Focus joints for this movement** (in order of importance): "
            f"{key_joints}\n"
        )
        if extreme_minors:
            priority_block += (
                f"**Other joints flagged only because their deviation is "
                f"large:** {extreme_minors}\n"
            )
    else:
        relevant_joints = list(per_joint.keys())
        priority_block = ""

    # Filter the data dumps to the joints we actually want the model to
    # reason about — keeps the prompt tight and the response on-topic.
    filtered_per_joint = {j: per_joint[j] for j in relevant_joints if j in per_joint}
    filtered_avg_diffs = {j: avg_diffs[j] for j in relevant_joints if j in avg_diffs}
    filtered_signed = {
        j: signed_diffs[j] for j in relevant_joints if j in signed_diffs
    }

    # Build a side-by-side angle table so Gemini can reason about specific
    # degrees rather than just "close" / "not close".
    angle_table_rows = []
    for j in relevant_joints:
        p = pro_angle_stats.get(j) or {}
        u = user_angle_stats.get(j) or {}
        if not p or not u:
            continue
        angle_table_rows.append(
            f"- {j}: pro mean {p['mean_deg']}° (range {p['min_deg']}–{p['max_deg']}°, "
            f"ROM {p['range_deg']}°) vs user mean {u['mean_deg']}° "
            f"(range {u['min_deg']}–{u['max_deg']}°, ROM {u['range_deg']}°)"
        )
    angle_table = "\n".join(angle_table_rows) or "(no per-joint angle stats available)"

    if merged_pairs:
        joined = ", ".join(merged_pairs)
        view_block = (
            f"\n**Merged joints:** The left and right copies of "
            f"**{joined}** overlap throughout both clips (the camera "
            f"angle puts them on top of each other), so each one is "
            f"reported as a single un-sided joint — e.g. just \"knee\" "
            f"rather than \"left_knee\" and \"right_knee\". Refer to "
            f"those joints without a left/right qualifier in your "
            f"response. Any joints that *do* have a left/right label "
            f"below are genuinely independent signals — feel free to "
            f"call out asymmetry between them if the data shows it.\n"
        )
    else:
        view_block = ""

    prompt = f"""You are an expert sports biomechanics coach analysing movement data.

A user is comparing their technique to a professional athlete for: **{movement_description}**

**Overall similarity score:** {overall:.1f}/100
{priority_block}{view_block}

**Per-joint similarity scores (0-100, higher = closer to pro):**
{json.dumps(filtered_per_joint, indent=2)}

**Mean angle differences per joint (degrees, absolute):**
{json.dumps(filtered_avg_diffs, indent=2)}

**Signed mean angle differences (user − pro, degrees).** Positive means the user's joint angle is *larger* than the pro's at the same moment in the movement (more extension / less flexion); negative means the user's angle is smaller (more flexion / less extension). Use this to tell the user which direction to correct.
{json.dumps(filtered_signed, indent=2)}

**Phase-level scores (0-100) — preparation, execution, follow-through:**
{json.dumps(phase_scores, indent=2)}

**Joint-by-joint angle profile (pro vs user, throughout the movement):**
{angle_table}

Write a focused markdown coaching response. Use your own judgement about what the data is actually saying — patterns across joints, timing of the divergence, ROM mismatches, anything that jumps out — and structure the response around the most useful observations. You don't need to cover every joint; the data above has been filtered to the ones worth discussing.

Cover, at minimum:
- A short summary of overall technique quality (2-3 sentences).
- The handful of highest-leverage corrections — quote the specific current vs target angles in degrees, explain the biomechanical reason the change matters, and give a concrete body cue the user can actually feel.
- One thing the user is doing well.
- A drill or focus for the next session that follows from the corrections.

Guidelines:
- Lean toward fewer, sharper points. If a joint isn't in the focus list and isn't an extreme outlier, don't mention it.
- Use the signed diffs to phrase corrections directionally — say "you under-rotate the shoulder by ~15°" or "your knee is over-extended by 8°", not "you differ by 12°".
- Use the phase scores to localise problems if they cluster — e.g. "your preparation is solid but your execution drops 20 points, so the issue is in the strike, not the setup".
- You're free to add observations that go beyond the listed sections if the data clearly supports them — e.g. "your knee and hip are both lagging the pro by ~15° at the same point in the movement, suggesting the issue is sequencing rather than any single joint".
- Be direct, plain English, no filler. Avoid jargon unless you define it.
- Do NOT mention the raw similarity scores — turn the numbers into coaching language.
"""

    response = _generate_with_retry(
        client,
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


def generate_insights_no_key(comparison_result: dict) -> str:
    # Rules-based fallback used when no Gemini key is set.
    per_joint = comparison_result.get("per_joint_scores", {})
    overall = comparison_result.get("overall_score", 0)
    joint_weights = comparison_result.get("joint_weights") or {}

    # Rank "areas to improve" by importance * deficit rather than just the
    # lowest raw score — otherwise we'd flag an irrelevant joint that
    # happened to score badly.
    def deficit(joint: str, score: float) -> float:
        w = joint_weights.get(joint, 1.0 / max(1, len(per_joint)))
        return w * (100.0 - score)

    sorted_joints = sorted(per_joint.items(), key=lambda kv: kv[1])
    weakest = sorted(
        per_joint.items(),
        key=lambda kv: deficit(kv[0], kv[1]),
        reverse=True,
    )[:3]
    strongest = sorted_joints[-1] if sorted_joints else None

    lines = []
    lines.append(f"### Overall Score: {overall:.1f}/100\n")

    if overall >= 85:
        lines.append("Your movement closely mirrors the professional. Focus on maintaining consistency.\n")
    elif overall >= 70:
        lines.append("Good technique overall with some areas for refinement.\n")
    elif overall >= 55:
        lines.append("Moderate similarity — there are clear areas to work on.\n")
    else:
        lines.append("Significant differences from the professional technique — consider working with a coach.\n")

    lines.append("\n### Areas to Improve\n")
    for joint, score in weakest:
        joint_label = joint.replace("_", " ").title()
        lines.append(f"- **{joint_label}**: Score {score:.0f}/100 — focus on matching the professional's angle through this range of motion.\n")

    if strongest:
        joint_label = strongest[0].replace("_", " ").title()
        lines.append(f"\n### Strength to Maintain\n- **{joint_label}**: Your best-matching joint — keep it consistent.\n")

    lines.append("\n*Add a Gemini API key in Settings to get personalised coaching insights.*")

    return "".join(lines)
