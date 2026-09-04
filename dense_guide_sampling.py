#!/usr/bin/env python3
"""Dense-guide one-dimensional and two-dimensional sampling.

This file implements the strategy discussed for WRS1D/WRS2D:

- For one score dimension, compute a dense-guide weight from the score
  distribution and rank samples by that weight.
- For two score dimensions, compute the dense-guide ranking independently for
  each dimension, sample a weighted without-replacement ranking, take the top
  ratio from each ranking, keep the intersection, and expand the ratio schedule
  until the intersection reaches the target budget.

The code is deliberately small and data-format agnostic. Scores are ordinary
floats, usually normalized into [0, 1].
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class DenseGuide:
    source_peak: float
    source_sigma: float
    target_mean: float
    target_sigma: float


@dataclass(frozen=True)
class WRS2DTrace:
    ratio: float
    top_k_each_dim: int
    intersection_size: int


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def write_records(path: Path, records: list[dict[str, Any]], jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def score_value(record: dict[str, Any], key: str) -> float:
    value = record[key]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty score list for key {key!r}")
        value = value[0]
    return float(value)


def parse_budget(total: int, ratio: float | None, count: int | None) -> int:
    if total <= 0:
        raise ValueError("No valid records")
    if count is not None:
        if count <= 0:
            raise ValueError("--count must be positive")
        return min(count, total)
    if ratio is None:
        raise ValueError("Provide either --ratio or --count")
    if not 0 < ratio <= 1:
        raise ValueError("--ratio must be in (0, 1]")
    return max(1, min(total, math.ceil(total * ratio)))


def gaussian_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), EPS)
    z = (x - mean) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def kde_peak(scores: np.ndarray, grid_size: int = 512) -> float:
    """Approximate the highest-density score value using a Gaussian KDE."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        raise ValueError("Empty score array")
    if np.allclose(scores, scores[0]):
        return float(scores[0])

    lo = float(scores.min())
    hi = float(scores.max())
    grid = np.linspace(lo, hi, grid_size)
    std = float(scores.std())
    bandwidth = 1.06 * std * (len(scores) ** (-1.0 / 5.0))
    bandwidth = max(bandwidth, (hi - lo) / max(grid_size - 1, 1), EPS)

    density = np.zeros_like(grid)
    chunk_size = 10000
    for start in range(0, len(scores), chunk_size):
        chunk = scores[start : start + chunk_size]
        diff = (grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * diff * diff).sum(axis=1)
    density /= len(scores) * bandwidth * math.sqrt(2.0 * math.pi)
    return float(grid[int(np.argmax(density))])


def dense_guide_weight(
    scores: np.ndarray,
    *,
    guide_strength: float = 0.5,
    target_sigma_scale: float = 1.0,
) -> tuple[np.ndarray, DenseGuide]:
    """Return dense-guide weights for one score dimension.

    `source_peak` is the densest point of the observed score distribution.
    The guide target is shifted from the source peak toward the maximum score:

        target_mean = source_peak + guide_strength * (max_score - source_peak)

    A sample's dense-guide weight is target_density / source_density. Ranking
    by this weight is the deterministic version of WRS1D used by the project.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        raise ValueError("Empty score array")

    source_sigma = max(float(scores.std()), EPS)
    if np.allclose(scores, scores[0]):
        weights = np.ones(len(scores), dtype=np.float64)
        value = float(scores[0])
        return weights, DenseGuide(value, source_sigma, value, source_sigma)

    source_peak = kde_peak(scores)
    max_score = float(scores.max())
    target_mean = source_peak + guide_strength * (max_score - source_peak)
    target_sigma = max(source_sigma * target_sigma_scale, EPS)

    source_density = gaussian_pdf(scores, source_peak, source_sigma)
    target_density = gaussian_pdf(scores, target_mean, target_sigma)
    weights = target_density / np.maximum(source_density, EPS)
    weights = np.nan_to_num(weights, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    return weights, DenseGuide(source_peak, source_sigma, target_mean, target_sigma)


def weighted_random_rank(weights: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Return a weighted random ordering without replacement.

    This uses the Efraimidis-Spirakis exponential-key trick. Items with larger
    weights tend to appear earlier, while low-weight items still have non-zero
    chance to be selected.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, np.finfo(np.float64).max)
    if float(weights.sum()) <= EPS:
        weights = np.ones(len(weights), dtype=np.float64)
    keys = -np.log(np.maximum(rng.random(len(weights)), EPS)) / np.maximum(weights, EPS)
    return np.argsort(keys).tolist()


def dense_rank(scores: np.ndarray, *, guide_strength: float = 0.5) -> tuple[list[int], np.ndarray, DenseGuide]:
    """Deterministic ranking by dense-guide weight, useful for visualization."""
    weights, guide = dense_guide_weight(scores, guide_strength=guide_strength)
    ranked = sorted(range(len(scores)), key=lambda i: (float(weights[i]), float(scores[i])), reverse=True)
    return ranked, weights, guide


def dense1d_select(
    scores: np.ndarray,
    budget: int,
    *,
    guide_strength: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, DenseGuide]:
    """Select `budget` indices using one-dimensional WRS over dense-guide weights."""
    weights, guide = dense_guide_weight(scores, guide_strength=guide_strength)
    ranked = weighted_random_rank(weights, np.random.default_rng(seed))
    return np.asarray(ranked[:budget], dtype=np.int64), weights, guide


def wrs2d_intersection_expand_select(
    text_scores: np.ndarray,
    clip_scores: np.ndarray,
    budget: int,
    *,
    expand_ratios: list[float] | tuple[float, ...] = (0.20, 0.24, 0.27, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00),
    guide_strength: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select by independent dense-guide WRS rankings plus expanding intersection."""
    text_scores = np.asarray(text_scores, dtype=np.float64)
    clip_scores = np.asarray(clip_scores, dtype=np.float64)
    if len(text_scores) != len(clip_scores):
        raise ValueError("text_scores and clip_scores must have the same length")
    if budget > len(text_scores):
        raise ValueError("budget cannot exceed the number of samples")

    text_weights, text_guide = dense_guide_weight(text_scores, guide_strength=guide_strength)
    clip_weights, clip_guide = dense_guide_weight(clip_scores, guide_strength=guide_strength)
    rng = np.random.default_rng(seed)
    text_ranked = weighted_random_rank(text_weights, rng)
    clip_ranked = weighted_random_rank(clip_weights, rng)
    text_rank_pos = {idx: rank for rank, idx in enumerate(text_ranked)}
    clip_rank_pos = {idx: rank for rank, idx in enumerate(clip_ranked)}

    ratios = sorted(set(float(ratio) for ratio in expand_ratios))
    if not ratios or ratios[-1] < 1.0:
        ratios.append(1.0)

    selected: list[int] = []
    used_ratio: float | None = None
    trace: list[WRS2DTrace] = []
    total = len(text_scores)

    for ratio in ratios:
        if not 0 < ratio <= 1:
            raise ValueError(f"Invalid expand ratio {ratio}; values must be in (0, 1]")
        k = max(1, min(total, math.ceil(total * ratio)))
        intersection = set(text_ranked[:k]) & set(clip_ranked[:k])
        ordered = sorted(
            intersection,
            key=lambda idx: (text_rank_pos[idx] + clip_rank_pos[idx], text_rank_pos[idx], clip_rank_pos[idx]),
        )
        trace.append(WRS2DTrace(ratio, k, len(ordered)))
        if len(ordered) >= budget:
            selected = ordered[:budget]
            used_ratio = ratio
            break

    if len(selected) < budget:
        raise RuntimeError("The final 100% intersection did not reach the requested budget")

    meta = {
        "method": "wrs2d_intersection_expand",
        "budget": budget,
        "used_ratio": used_ratio,
        "expand_trace": [asdict(item) for item in trace],
        "text_guide": asdict(text_guide),
        "clip_guide": asdict(clip_guide),
        "text_weight_minmax": [float(text_weights.min()), float(text_weights.max())],
        "clip_weight_minmax": [float(clip_weights.min()), float(clip_weights.max())],
    }
    return np.asarray(selected, dtype=np.int64), meta


def parse_ratios(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=["dense1d", "wrs2d"])
    parser.add_argument("--score-key", default="yes_target_logprob_7B_NImg")
    parser.add_argument("--clip-key", default="clip_score")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--expand-ratios", default="0.20,0.24,0.27,0.30,0.35,0.40,0.50,0.60,0.80,1.00")
    parser.add_argument("--guide-strength", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--meta-output", type=Path, default=None)
    args = parser.parse_args()

    records = read_records(args.input)
    score_keys = [args.score_key] if args.method == "dense1d" else [args.score_key, args.clip_key]
    filtered = []
    for item in records:
        try:
            for key in score_keys:
                score_value(item, key)
            filtered.append(item)
        except Exception:
            pass

    budget = parse_budget(len(filtered), args.ratio, args.count)
    text_scores = np.array([score_value(item, args.score_key) for item in filtered], dtype=np.float64)

    if args.method == "dense1d":
        selected_idx, weights, guide = dense1d_select(
            text_scores,
            budget,
            guide_strength=args.guide_strength,
            seed=args.seed,
        )
        meta = {
            "method": "dense1d",
            "budget": budget,
            "score_key": args.score_key,
            "guide": asdict(guide),
            "weight_minmax": [float(weights.min()), float(weights.max())],
        }
    else:
        clip_scores = np.array([score_value(item, args.clip_key) for item in filtered], dtype=np.float64)
        selected_idx, meta = wrs2d_intersection_expand_select(
            text_scores,
            clip_scores,
            budget,
            expand_ratios=parse_ratios(args.expand_ratios),
            guide_strength=args.guide_strength,
            seed=args.seed,
        )

    selected = [filtered[int(idx)] for idx in selected_idx]
    write_records(args.output, selected, jsonl=args.jsonl)
    if args.meta_output is not None:
        args.meta_output.parent.mkdir(parents=True, exist_ok=True)
        args.meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"selected {len(selected)} / {len(filtered)} records")


if __name__ == "__main__":
    main()
