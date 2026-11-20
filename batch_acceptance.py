"""Freeze frame-level audit samples and assess a finite batch."""

from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
from math import comb, isfinite
from random import Random


AUTO_CANDIDATE = "auto_candidate"
HELD = "held"
HUMAN_CONFIRMED = "human_confirmed"
STRATA = (AUTO_CANDIDATE, HELD, HUMAN_CONFIRMED)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _validate_count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_rate(value, name, *, allow_endpoints):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not isfinite(value):
        raise ValueError(f"{name} must be numeric and between 0 and 1")
    if not allow_endpoints and value in (0, 1):
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return Fraction(str(value))


def _normalize_frames(frames):
    normalized = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("each frame must provide frame_id, stratum and revision")
        frame_id = frame.get("frame_id")
        revision = frame.get("revision")
        stratum = frame.get("stratum")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be nonempty")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"frame revision is missing for {frame_id}")
        if stratum not in STRATA:
            raise ValueError(f"unknown stratum for {frame_id}: {stratum!r}")
        if frame_id in normalized:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        normalized[frame_id] = {"frame_id": frame_id, "stratum": stratum, "revision": revision}
    return [normalized[frame_id] for frame_id in sorted(normalized)]


def _normalize_policy(policy):
    if policy is None:
        return None
    if not isinstance(policy, Mapping) or set(policy) != {"revision", "max_error_rate", "confidence"}:
        raise ValueError("policy must specify revision, max_error_rate and confidence")
    if not isinstance(policy["revision"], str) or not policy["revision"].strip():
        raise ValueError("policy revision must be nonempty")
    _validate_rate(policy["max_error_rate"], "max_error_rate", allow_endpoints=True)
    _validate_rate(policy["confidence"], "confidence", allow_endpoints=False)
    return dict(policy)


def freeze_batch(frames, *, sample_sizes, seed, policy=None):
    """Bind the candidate population, predeclared sampling plan and policy.

    Each item supplies a stable frame_id, disposition stratum, and revision
    binding the image and prior annotation state. Policy is None until approved.
    """
    _validate_count(seed, "seed")
    if not isinstance(sample_sizes, Mapping) or set(sample_sizes) != set(STRATA):
        raise ValueError("sample_sizes must specify each audit stratum")
    sizes = {stratum: _validate_count(sample_sizes[stratum], stratum) for stratum in STRATA}
    frames = _normalize_frames(frames)
    population = {stratum: [] for stratum in STRATA}
    for frame in frames:
        population[frame["stratum"]].append(frame["frame_id"])
    sample_ids = {}
    for stratum, members in population.items():
        rng = Random(f"batch-acceptance-v1:{seed}:{stratum}")
        sample_ids[stratum] = sorted(rng.sample(members, min(sizes[stratum], len(members))))
    manifest = {
        "schema": 1,
        "frames": frames,
        "sample_ids": sample_ids,
        "sample_sizes": sizes,
        "seed": seed,
        "policy": _normalize_policy(policy),
    }
    return {**manifest, "sha256": hashlib.sha256(_canonical(manifest).encode("utf-8")).hexdigest()}


def hypergeometric_error_bounds(population, sample_size, errors, confidence):
    """Return exact, separate one-sided lower and upper error-count bounds.

    The sample is drawn uniformly without replacement from a fixed population.
    Tests are inverted with inclusive tail probabilities, conservatively keeping
    totals whose probability is exactly the one-sided significance level.
    """
    _validate_count(population, "population")
    _validate_count(sample_size, "sample_size")
    _validate_count(errors, "errors")
    level = _validate_rate(confidence, "confidence", allow_endpoints=False)
    if not errors <= sample_size <= population:
        raise ValueError("expected errors <= sample_size <= population")
    if sample_size == population:
        return errors, errors
    if sample_size == 0:
        return 0, population

    alpha = 1 - level
    tail_target = alpha.numerator * comb(population, sample_size)
    feasible_max = population - sample_size + errors

    def accepted(total, upper):
        smallest = max(0, sample_size - (population - total))
        largest = min(sample_size, total)
        start, stop = (smallest, errors) if upper else (errors, largest)
        term = comb(total, start) * comb(population - total, sample_size - start)
        favorable = term
        for observed in range(start, stop):
            term = (
                term * (total - observed) * (sample_size - observed)
                // ((observed + 1) * (population - total - sample_size + observed + 1))
            )
            favorable += term
        return favorable * alpha.denominator >= tail_target

    # The lower-tail probability decreases as the unknown number of errors grows.
    lo, hi = errors, feasible_max
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if accepted(middle, upper=True):
            lo = middle
        else:
            hi = middle - 1
    upper_bound = lo

    # The upper-tail probability increases as the unknown number of errors grows.
    lo, hi = errors, feasible_max
    while lo < hi:
        middle = (lo + hi) // 2
        if accepted(middle, upper=False):
            hi = middle
        else:
            lo = middle + 1
    return lo, upper_bound


def _verified_manifest(manifest, current_frames):
    required = {"schema", "frames", "sample_ids", "sample_sizes", "seed", "policy", "sha256"}
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ValueError("invalid batch audit manifest")
    # Recreate the expected sample membership, not just the saved digest.
    expected = freeze_batch(
        manifest["frames"],
        sample_sizes=manifest["sample_sizes"],
        seed=manifest["seed"],
        policy=manifest["policy"],
    )
    if _canonical(manifest) != _canonical(expected):
        raise ValueError("batch audit manifest or sample membership has changed")
    if current_frames is not None and _normalize_frames(current_frames) != expected["frames"]:
        raise ValueError("frame membership, stratum or revision changed after the audit freeze")
    return expected


def assess_batch(manifest, reviews, *, current_frames=None):
    """Score sampled blind whole-frame reviews without modifying the batch.

    True denotes at least one material error. Missing reviews only indicate
    pending work. Reviews for unsampled frames are rejected.
    """
    frozen = _verified_manifest(manifest, current_frames)
    expected_ids = {frame_id for members in frozen["sample_ids"].values() for frame_id in members}
    if not isinstance(reviews, Mapping) or any(frame_id not in expected_ids for frame_id in reviews):
        raise ValueError("audit reviews must refer to frames selected in the frozen manifest")
    if any(type(is_error) is not bool for is_error in reviews.values()):
        raise ValueError("each completed audit review must state whether the frame has an error")

    counts = {stratum: 0 for stratum in STRATA}
    for frame in frozen["frames"]:
        counts[frame["stratum"]] += 1
    policy = frozen["policy"]
    summaries = {}
    for stratum, population in counts.items():
        planned = frozen["sample_ids"][stratum]
        checked = [reviews[frame_id] for frame_id in planned if frame_id in reviews]
        sample_size = len(planned)
        errors = sum(checked)
        summary = {
            "population": population,
            "sample_size": sample_size,
            "reviewed": len(checked),
            "errors": errors,
            "inclusion_probability": sample_size / population if population else None,
            "lower_errors": None,
            "upper_errors": None,
            "lower_bound": None,
            "upper_bound": None,
        }
        if not population:
            summary["status"] = "not_applicable"
        elif not sample_size:
            summary["status"] = "not_sampled"
        elif len(checked) < sample_size:
            summary["status"] = "pending_reviews"
        else:
            summary["status"] = "complete"
            if policy is not None:
                lower, upper = hypergeometric_error_bounds(population, sample_size, errors, policy["confidence"])
                summary.update(lower_errors=lower, upper_errors=upper, lower_bound=lower / population,
                               upper_bound=upper / population)
        summaries[stratum] = summary

    auto = summaries[AUTO_CANDIDATE]
    if policy is None:
        status = "pending_policy"
    elif any(summary["status"] == "pending_reviews" for summary in summaries.values()):
        status = "pending_reviews"
    elif not auto["population"]:
        status = "passed" if summaries[HUMAN_CONFIRMED]["population"] else "insufficient_evidence"
    elif auto["status"] != "complete":
        status = "insufficient_evidence"
    else:
        limit = Fraction(str(policy["max_error_rate"]))
        if Fraction(auto["upper_errors"], auto["population"]) <= limit:
            status = "passed"
        elif Fraction(auto["lower_errors"], auto["population"]) > limit:
            status = "exceeds_limit"
        else:
            status = "insufficient_evidence"

    return {
        "status": status,
        "eligible": status == "passed",
        "manifest_sha256": frozen["sha256"],
        "policy": policy,
        "strata": summaries,
    }
