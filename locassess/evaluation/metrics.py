"""Metric functions for the evaluation (SPEC §13). Pure functions over plain data."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

NAN = float("nan")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def binary_prf(gold: Sequence[int], pred: Sequence[int]) -> dict:
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if not g and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    tn = sum(1 for g, p in zip(gold, pred) if not g and not p)
    precision = tp / (tp + fp) if tp + fp else NAN
    recall = tp / (tp + fn) if tp + fn else NAN
    if tp:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if fp + fn else NAN      # no positives anywhere → undefined
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa for two raters; NaN if undefined (no items or no variation at all)."""
    n = len(a)
    if n == 0:
        return NAN
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return NAN if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(ratings: Sequence[Sequence]) -> float:
    """Fleiss' kappa; `ratings[i]` are the labels the m raters gave item i (same m for all items)."""
    if not ratings:
        return NAN
    m = len(ratings[0])
    if m < 2 or any(len(r) != m for r in ratings):
        return NAN
    cats = sorted({x for r in ratings for x in r}, key=str)
    counts = np.array([[Counter(r)[c] for c in cats] for r in ratings], dtype=float)
    n_items = len(ratings)
    p_j = counts.sum(axis=0) / (n_items * m)
    p_i = ((counts ** 2).sum(axis=1) - m) / (m * (m - 1))
    p_bar, pe = p_i.mean(), (p_j ** 2).sum()
    return NAN if pe == 1 else float((p_bar - pe) / (1 - pe))


def share_identical(ratings: Sequence[Sequence]) -> float:
    """Share of items on which all raters / repetitions gave the same label."""
    if not ratings:
        return NAN
    return sum(1 for r in ratings if len(set(r)) == 1) / len(ratings)


def brier(probs: Sequence[float], gold: Sequence[int]) -> float:
    if not probs:
        return NAN
    return float(np.mean([(p - g) ** 2 for p, g in zip(probs, gold)]))


def average_precision(probs: Sequence[float], gold: Sequence[int]) -> float:
    """PR-AUC as average precision; NaN if there is no positive case."""
    if not probs or not any(gold):
        return NAN
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(list(gold), list(probs)))


def reliability_bins(probs: Sequence[float], gold: Sequence[int], n_bins: int = 10) -> list[dict]:
    """Data for a reliability diagram: per probability bin the mean prediction and observed frequency."""
    out = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == n_bins - 1 and p == 1.0)]
        out.append({"bin": b, "lower": lo, "upper": hi, "count": len(idx),
                    "mean_predicted": float(np.mean([probs[i] for i in idx])) if idx else NAN,
                    "observed_frequency": float(np.mean([gold[i] for i in idx])) if idx else NAN})
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return NAN
    from scipy.stats import spearmanr

    if len(set(a)) == 1 or len(set(b)) == 1:
        return NAN
    return float(spearmanr(a, b).statistic)


def normalise_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().casefold()


def quote_in_text(quote: str, text: str) -> bool:
    """Verbatim check (whitespace and case normalised, curly quotes unified)."""
    def norm(s: str) -> str:
        return normalise_text(s.replace("„", '"').replace("“", '"').replace("”", '"').replace("’", "'"))

    q = norm(quote).strip(" .\"'")
    return bool(q) and q in norm(text)


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return NAN, NAN
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
