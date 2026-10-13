"""Evaluation runner (SPEC §13): loads data from the database, computes the metrics and writes
one CSV per metric to results/. Metrics without the necessary data are skipped with a reason.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

from locassess.config import AppConfig
from locassess.db import repo
from locassess.evaluation import metrics as M
from locassess.stage0_corpus import normalize_url
from locassess.stage2_score import Evidence, ScoringParams, build_profile, overall_score

NAN = float("nan")


class Skipped(Exception):
    """Raised by a metric when the data it needs is not available."""


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _main_ids(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT main_id FROM criterion ORDER BY main_id")]


def _factors(conn) -> dict[str, str]:
    return repo.criteria_factors(conn)


def _location(conn, corpus_id: int) -> str:
    return (json.loads(repo.get_corpus(conn, corpus_id)["location_json"]).get("name") or "").strip()


def _keys(value) -> list[str]:
    if not value:
        return []
    try:
        v = json.loads(value)
        if isinstance(v, list):
            return [str(x) for x in v]
    except (json.JSONDecodeError, TypeError):
        pass
    return [k.strip() for k in str(value).replace(";", ",").split(",") if k.strip()]


def model_cells(conn, extraction_run_id: int) -> dict[tuple[int, str], dict]:
    """article × main criterion for one extraction run: evidence fields and P(affected = yes)."""
    run = repo.get_extraction_run(conn, extraction_run_id)
    mains = _main_ids(conn)
    cells = {(a["article_id"], m): {"affected": 0, "p_yes": None}
             for a in repo.get_articles(conn, run["corpus_id"], selected_only=True) for m in mains}
    for r in conn.execute("SELECT article_id, distributions_json FROM model_response "
                          "WHERE extraction_run_id = ? AND query_type = 'A'", (extraction_run_id,)):
        dist = json.loads(r["distributions_json"])["distributions"]
        for m in mains:
            p = dist.get(f"{m}_affected", {}).get("yes")
            if p is not None and (r["article_id"], m) in cells:
                c = cells[(r["article_id"], m)]
                c["p_yes"] = p if c["p_yes"] is None else max(c["p_yes"], p)   # sections: highest P(yes)
    for e in conn.execute("SELECT * FROM evidence WHERE extraction_run_id = ?", (extraction_run_id,)):
        c = cells.setdefault((e["article_id"], e["main_id"]), {"p_yes": None})
        c.update(affected=1, direction=e["direction"], intensity=e["intensity"], type=e["evidence_type"],
                 sub_id=e["sub_id"], sentence_keys=_keys(e["sentence_keys"]))
    return cells


def gold_cells(conn, corpus_id: int) -> dict[str, dict[tuple[int, str], dict]]:
    """annotator → article × criterion. Unannotated criteria of an annotated article count as not affected."""
    articles = {a["article_id"] for a in repo.get_articles(conn, corpus_id, selected_only=True)}
    mains = _main_ids(conn)
    out: dict[str, dict] = defaultdict(dict)
    for r in conn.execute("SELECT * FROM annotation"):
        if r["article_id"] in articles:
            out[r["annotator"]][(r["article_id"], r["main_id"])] = {
                "affected": r["affected"], "direction": r["direction"], "intensity": r["intensity"],
                "type": r["evidence_type"], "sub_id": r["sub_id"], "sentence_keys": _keys(r["sentence_keys"])}
    for cells in out.values():
        for a in {k[0] for k in cells}:
            for m in mains:
                cells.setdefault((a, m), {"affected": 0})
    return dict(out)


def evidence_of(conn, extraction_run_id: int) -> list[Evidence]:
    return repo.load_evidence(conn, extraction_run_id)


def default_params(conn, cfg: AppConfig, extraction_run_id: int, scoring=None) -> ScoringParams:
    period_from, period_to = repo.extraction_run_period(conn, extraction_run_id)
    return ScoringParams.from_config(scoring or cfg.scoring, run_date=period_to or cfg.resolved_run_date(),
                                     period_from=period_from or period_to)


def hybrid_scores(conn, cfg: AppConfig, extraction_run_id: int) -> dict[str, tuple[float | None, str | None, str]]:
    """main_id → (score, rating, status) of an extraction run under the default scoring parameters."""
    p = build_profile(evidence_of(conn, extraction_run_id), _main_ids(conn), default_params(conn, cfg, extraction_run_id))
    return {k: (c.score, c.rating, c.status) for k, c in p.criteria.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def search_reproducibility(conn, cfg) -> list[dict]:
    corp = [r["corpus_id"] for r in conn.execute("SELECT corpus_id FROM corpus ORDER BY corpus_id")]
    urls = {c: {normalize_url(a["url"]) for a in repo.get_articles(conn, c, selected_only=True)} for c in corp}
    rows = []
    for a, b in itertools.combinations(corp, 2):
        if _location(conn, a).casefold() != _location(conn, b).casefold() or not urls[a] or not urls[b]:
            continue
        rows.append({"corpus_a": a, "corpus_b": b, "location": _location(conn, a),
                     "created_a": repo.get_corpus(conn, a)["created_at"], "created_b": repo.get_corpus(conn, b)["created_at"],
                     "selected_a": len(urls[a]), "selected_b": len(urls[b]), "overlap": len(urls[a] & urls[b]),
                     "jaccard": M.jaccard(urls[a], urls[b])})
    if not rows:
        raise Skipped("needs two corpora with selected articles for the same location (run Stage 0 twice)")
    return rows


def _annotated_runs(conn):
    for r in conn.execute("SELECT * FROM extraction_run ORDER BY extraction_run_id"):
        gold = gold_cells(conn, r["corpus_id"])
        if gold:
            yield r, gold


def extraction_quality(conn, cfg) -> tuple[list[dict], list[dict]]:
    rows, by_crit = [], []
    for run, gold in _annotated_runs(conn):
        pred = model_cells(conn, run["extraction_run_id"])
        for annotator, g in gold.items():
            keys = sorted(k for k in g if k in pred)
            ga, pa = [g[k]["affected"] for k in keys], [pred[k]["affected"] for k in keys]
            both = [k for k in keys if g[k]["affected"] and pred[k]["affected"]]

            def agree(field):
                pairs = [(g[k].get(field), pred[k].get(field)) for k in both]
                return (sum(1 for x, y in pairs if x == y) / len(pairs)) if pairs else NAN

            sent = [bool(set(pred[k].get("sentence_keys") or []) & set(g[k].get("sentence_keys") or []))
                    for k in both if g[k].get("sentence_keys")]
            prf = M.binary_prf(ga, pa)
            rows.append({
                "extraction_run_id": run["extraction_run_id"], "corpus_id": run["corpus_id"], "model": run["model"],
                "repetition": run["repetition"], "annotator": annotator, "articles": len({k[0] for k in keys}),
                "cells": len(keys), **prf, "kappa_affected": M.cohen_kappa(ga, pa), "both_affected": len(both),
                "direction_agreement": agree("direction"),
                "direction_kappa": M.cohen_kappa([g[k]["direction"] for k in both], [pred[k]["direction"] for k in both]),
                "intensity_agreement": agree("intensity"), "evidence_type_agreement": agree("type"),
                "sub_criterion_agreement": agree("sub_id"),
                "sentence_overlap": (sum(sent) / len(sent)) if sent else NAN,
            })
            for m in _main_ids(conn):
                mk = [k for k in keys if k[1] == m]
                p = M.binary_prf([g[k]["affected"] for k in mk], [pred[k]["affected"] for k in mk])
                by_crit.append({"extraction_run_id": run["extraction_run_id"], "model": run["model"],
                                "annotator": annotator, "main_id": m, "gold_positive": p["tp"] + p["fn"],
                                "tp": p["tp"], "fp": p["fp"], "fn": p["fn"]})
    if not rows:
        raise Skipped("needs manual annotations (locassess annotate template / import)")
    return rows, by_crit


def invalid_evidence(conn, extraction_run_id: int) -> tuple[list[sqlite3.Row], set[int]]:
    """All evidence of a run and the IDs whose sentences do not exist in the article (or differ from it)."""
    ev = conn.execute("SELECT * FROM evidence WHERE extraction_run_id = ?", (extraction_run_id,)).fetchall()
    bad = set()
    for e in ev:
        texts = {s["sentence_key"]: s["text"] for s in repo.get_sentences(conn, e["article_id"])}
        keys = _keys(e["sentence_keys"])
        if not keys or any(k not in texts for k in keys) or \
                M.normalise_text(" ".join(texts[k] for k in keys)) != M.normalise_text(e["sentence_text"]):
            bad.add(e["evidence_id"])
    return ev, bad


def hallucination(conn, cfg) -> list[dict]:
    rows = []
    for run in conn.execute("SELECT * FROM extraction_run ORDER BY extraction_run_id"):
        ev, bad_ids = invalid_evidence(conn, run["extraction_run_id"])
        bad = len(bad_ids)
        rows.append({"source": "hybrid", "id": run["extraction_run_id"], "corpus_id": run["corpus_id"],
                     "model": run["model"], "repetition": run["repetition"], "statements": len(ev),
                     "not_in_article": bad, "rate": bad / len(ev) if ev else NAN})
    for b in conn.execute("SELECT * FROM baseline_assessment ORDER BY baseline_id"):
        q, bad = _baseline_quotes(conn, b)
        rows.append({"source": "baseline", "id": b["baseline_id"], "corpus_id": b["corpus_id"], "model": b["model"],
                     "repetition": b["repetition"], "statements": q, "not_in_article": bad,
                     "rate": bad / q if q else NAN})
    if not rows:
        raise Skipped("no extraction runs or baseline assessments")
    return rows


def _baseline_texts(conn, b) -> dict[str, str]:
    labels = json.loads(b["overall_json"]).get("article_labels", {})
    texts = {a["article_id"]: a["text"] for a in repo.get_articles(conn, b["corpus_id"], selected_only=True)}
    return {label: texts.get(aid, "") for label, aid in labels.items()}


def _baseline_quotes(conn, b) -> tuple[int, int]:
    texts = _baseline_texts(conn, b)
    n = bad = 0
    for c in json.loads(b["per_criterion_json"]).values():
        for e in c.get("evidence") or []:
            n += 1
            if not M.quote_in_text(e.get("quote", ""), texts.get(e.get("article"), "")):
                bad += 1
    return n, bad


def extraction_stability(conn, cfg) -> list[dict]:
    groups = defaultdict(list)
    for r in conn.execute("SELECT * FROM extraction_run ORDER BY extraction_run_id"):
        groups[(r["corpus_id"], r["model"], r["query_definition_version"])].append(r["extraction_run_id"])
    rows = []
    for (corpus_id, model, qdv), runs in groups.items():
        if len(runs) < 2:
            continue
        cells = [model_cells(conn, r) for r in runs]
        keys = sorted(cells[0])
        affected = [[c[k]["affected"] for c in cells] for k in keys]
        all_aff = [k for k in keys if all(c[k]["affected"] for c in cells)]
        direction = [[c[k]["direction"] for c in cells] for k in all_aff]
        scores = [hybrid_scores(conn, cfg, r) for r in runs]
        ratings = [[s[m][1] or "none" for s in scores] for m in _main_ids(conn)]
        rows.append({"corpus_id": corpus_id, "model": model, "query_definition_version": qdv, "runs": len(runs),
                     "cells": len(keys), "share_identical_affected": M.share_identical(affected),
                     "fleiss_kappa_affected": M.fleiss_kappa(affected), "cells_affected_in_all_runs": len(all_aff),
                     "share_identical_direction": M.share_identical(direction),
                     "fleiss_kappa_direction": M.fleiss_kappa(direction),
                     "share_identical_rating": M.share_identical(ratings)})
    if not rows:
        raise Skipped("needs at least 2 repetitions of one model on one corpus (locassess extract --repeat 3)")
    return rows


def calibration(conn, cfg) -> tuple[list[dict], list[dict]]:
    rows, bins = [], []
    for run, gold in _annotated_runs(conn):
        pred = model_cells(conn, run["extraction_run_id"])
        for annotator, g in gold.items():
            keys = [k for k in sorted(g) if k in pred and pred[k]["p_yes"] is not None]
            probs, y = [pred[k]["p_yes"] for k in keys], [g[k]["affected"] for k in keys]
            rows.append({"extraction_run_id": run["extraction_run_id"], "model": run["model"],
                         "repetition": run["repetition"], "confidence_method": run["confidence_method"],
                         "annotator": annotator, "cells": len(keys), "positives": sum(y),
                         "brier": M.brier(probs, y), "pr_auc": M.average_precision(probs, y)})
            bins += [{"extraction_run_id": run["extraction_run_id"], "model": run["model"], "annotator": annotator, **b}
                     for b in M.reliability_bins(probs, y)]
    if not rows:
        raise Skipped("needs manual annotations (locassess annotate template / import)")
    return rows, bins


PARAM_GRID = {
    "type_weights": {"default": None, "equal": {"fact": 1.0, "announcement": 1.0, "opinion": 1.0},
                     "facts_heavy": {"fact": 1.0, "announcement": 0.3, "opinion": 0.1}},
    "half_life_months": {"default": None, "12": 12, "36": 36, "no_decay": 1e9},
    "thresholds": {"default": None, "0.75": 0.75, "1.25": 1.25},
}


def _variants(base: ScoringParams):
    for (tw_name, tw), (hl_name, hl), (th_name, th) in itertools.product(
            PARAM_GRID["type_weights"].items(), PARAM_GRID["half_life_months"].items(), PARAM_GRID["thresholds"].items()):
        p = base
        if tw:
            p = replace(p, type_weights=tw)
        if hl:
            p = replace(p, half_life_months=hl)
        if th:
            p = replace(p, green_threshold=th, red_threshold=-th)
        yield f"tw={tw_name}|hl={hl_name}|thr={th_name}", p


def scoring_robustness(conn, cfg, n_draws: int = 1000, seed: int = 42) -> tuple[list[dict], list[dict]]:
    mains, factors = _main_ids(conn), _factors(conn)
    runs = conn.execute("SELECT * FROM extraction_run ORDER BY extraction_run_id").fetchall()
    if not runs:
        raise Skipped("no extraction runs")
    variant_rows, ranking_rows = [], []
    base_overall: dict[int, dict[str, float | None]] = {}
    for run in runs:
        ev, base = evidence_of(conn, run["extraction_run_id"]), default_params(conn, cfg, run["extraction_run_id"])
        ref = build_profile(ev, mains, base, factors=factors)
        rated = [m for m in mains if ref.criteria[m].rating]
        base_overall[run["extraction_run_id"]] = {}
        for name, p in _variants(base):
            prof = build_profile(ev, mains, p, factors=factors)
            same = sum(1 for m in rated if prof.criteria[m].rating == ref.criteria[m].rating)
            base_overall[run["extraction_run_id"]][name] = prof.overall.score
            variant_rows.append({"extraction_run_id": run["extraction_run_id"], "corpus_id": run["corpus_id"],
                                 "model": run["model"], "variant": name, "rated_criteria": len(rated),
                                 "changed_ratings": len(rated) - same,
                                 "share_same_rating": same / len(rated) if rated else NAN,
                                 "overall": prof.overall.score})

    # ranking across locations: latest extraction run per location (by model)
    latest: dict[tuple[str, str], sqlite3.Row] = {}
    for run in runs:
        latest[(_location(conn, run["corpus_id"]).casefold(), run["model"])] = run
    by_model = defaultdict(list)
    for (loc, model), run in latest.items():
        by_model[model].append(run)
    rng = random.Random(seed)
    for model, locs in by_model.items():
        if len(locs) < 2:
            continue
        ids = [r["extraction_run_id"] for r in locs]
        scores = {i: {m: c.score for m, c in build_profile(evidence_of(conn, i), mains,
                                                             default_params(conn, cfg, i)).criteria.items()} for i in ids}
        equal = [overall_score(scores[i]).score for i in ids]
        rhos, same_top = [], 0
        for _ in range(n_draws):
            sl = {m: rng.randint(0, 10) for m in mains}
            g = [overall_score(scores[i], sl).score for i in ids]
            g = [x if x is not None else -99 for x in g]
            rhos.append(M.spearman([x if x is not None else -99 for x in equal], g))
            same_top += int(max(range(len(ids)), key=lambda j: g[j]) ==
                            max(range(len(ids)), key=lambda j: equal[j] if equal[j] is not None else -99))
        valid = [r for r in rhos if r == r]
        variant_rho = [M.spearman([base_overall[i]["tw=default|hl=default|thr=default"] or -99 for i in ids],
                                  [base_overall[i][v] or -99 for i in ids]) for v in base_overall[ids[0]]]
        ranking_rows.append({"model": model, "locations": len(ids), "extraction_runs": " ".join(map(str, ids)),
                             "slider_draws": n_draws, "seed": seed,
                             "spearman_mean": statistics.mean(valid) if valid else NAN,
                             "spearman_min": min(valid) if valid else NAN,
                             "share_same_top_location": same_top / n_draws,
                             "param_variants_spearman_mean": statistics.mean([r for r in variant_rho if r == r])
                             if any(r == r for r in variant_rho) else NAN})
    return variant_rows, ranking_rows


def baseline_comparison(conn, cfg) -> tuple[list[dict], list[dict]]:
    mains = _main_ids(conn)
    detail, summary = [], []
    corpora = [r[0] for r in conn.execute("SELECT DISTINCT corpus_id FROM baseline_assessment ORDER BY corpus_id")]
    for corpus_id in corpora:
        base_rows = conn.execute("SELECT * FROM baseline_assessment WHERE corpus_id = ?", (corpus_id,)).fetchall()
        base = [json.loads(b["per_criterion_json"]) for b in base_rows]
        base_backed, base_rated = 0, 0
        for b, data in zip(base_rows, base):
            texts = _baseline_texts(conn, b)
            for c in data.values():
                if c.get("score") is not None:
                    base_rated += 1
                    base_backed += any(M.quote_in_text(e.get("quote", ""), texts.get(e.get("article"), ""))
                                       for e in c.get("evidence") or [])
        models = defaultdict(list)
        for r in conn.execute("SELECT * FROM extraction_run WHERE corpus_id = ?", (corpus_id,)):
            models[r["model"]].append(r["extraction_run_id"])
        for model, runs in models.items():
            hyb = [hybrid_scores(conn, cfg, r) for r in runs]
            hyb_rated = hyb_backed = 0
            for r, h in zip(runs, hyb):   # a rated criterion is backed if one of its evidence sentences is verified
                ev, bad = invalid_evidence(conn, r)
                ok: dict[str, bool] = defaultdict(bool)
                for e in ev:
                    ok[e["main_id"]] |= e["evidence_id"] not in bad
                rated = [m for m in mains if h[m][0] is not None]
                hyb_rated += len(rated)
                hyb_backed += sum(1 for m in rated if ok[m])
            sd_h, sd_b, modal = [], [], []
            for m in mains:
                hs = [h[m][0] for h in hyb if h[m][0] is not None]
                bs = [d[m]["score"] for d in base if d.get(m, {}).get("score") is not None]
                hm, hsd = M.mean_sd(hs)
                bm, bsd = M.mean_sd(bs)
                hr = Counter(h[m][1] for h in hyb if h[m][1]).most_common(1)
                br = Counter(d[m]["rating"] for d in base if d.get(m, {}).get("rating")).most_common(1)
                detail.append({"corpus_id": corpus_id, "hybrid_model": model, "main_id": m,
                               "hybrid_runs": len(hyb), "hybrid_rated": len(hs), "hybrid_mean": hm, "hybrid_sd": hsd,
                               "hybrid_rating": hr[0][0] if hr else "", "baseline_reps": len(base),
                               "baseline_rated": len(bs), "baseline_mean": bm, "baseline_sd": bsd,
                               "baseline_rating": br[0][0] if br else ""})
                if len(hs) > 1:
                    sd_h.append(hsd)
                if len(bs) > 1:
                    sd_b.append(bsd)
                if hr and br:
                    modal.append((hr[0][0], br[0][0]))
            summary.append({
                "corpus_id": corpus_id, "hybrid_model": model, "hybrid_runs": len(hyb), "baseline_reps": len(base),
                "mean_sd_hybrid": statistics.mean(sd_h) if sd_h else NAN,
                "mean_sd_baseline": statistics.mean(sd_b) if sd_b else NAN,
                "criteria_rated_hybrid_mean": hyb_rated / len(hyb) if hyb else NAN,
                "criteria_rated_baseline_mean": base_rated / len(base) if base else NAN,
                "share_backed_hybrid": hyb_backed / hyb_rated if hyb_rated else NAN,
                "share_backed_baseline": base_backed / base_rated if base_rated else NAN,
                "criteria_rated_by_both": len(modal),
                "rating_agreement": (sum(1 for a, b in modal if a == b) / len(modal)) if modal else NAN,
                "rating_kappa": M.cohen_kappa([a for a, _ in modal], [b for _, b in modal]),
            })
    if not summary:
        raise Skipped("needs baseline assessments (locassess baseline --corpus N --repeat 3)")
    return detail, summary


def report_reliability(conn, cfg) -> tuple[list[dict], list[dict]]:
    reports = conn.execute("SELECT * FROM report ORDER BY report_id").fetchall()
    if not reports:
        raise Skipped("no reports")
    per_report, rules = [], defaultdict(lambda: {"checked": 0, "failed": 0, "first_checked": 0, "first_failed": 0})
    for r in reports:
        attempts = json.loads(r["checks_json"] or "[]")
        per_report.append({"report_id": r["report_id"], "scoring_run_id": r["scoring_run_id"], "passed": r["passed"],
                           "attempts": r["attempts"], "prompt_version": r["prompt_version"]})
        for a in attempts:
            for c in a["checks"]:
                s = rules[c["rule"]]
                s["checked"] += 1
                s["failed"] += int(not c["passed"])
                if a["attempt"] == 1:
                    s["first_checked"] += 1
                    s["first_failed"] += int(not c["passed"])
    summary = [{"rule": "(all reports)", "reports": len(reports),
                "pass_rate": sum(r["passed"] for r in reports) / len(reports),
                "mean_attempts": statistics.mean(r["attempts"] for r in reports),
                "failure_rate_all_attempts": NAN, "failure_rate_first_attempt": NAN}]
    for rule, s in sorted(rules.items()):
        summary.append({"rule": rule, "reports": len(reports), "pass_rate": NAN, "mean_attempts": NAN,
                        "failure_rate_all_attempts": s["failed"] / s["checked"] if s["checked"] else NAN,
                        "failure_rate_first_attempt": s["first_failed"] / s["first_checked"] if s["first_checked"] else NAN})
    return per_report, summary


def _cost(row_model: str | None, tin, tout, cost, prices) -> float:
    if cost is not None:
        return cost
    p = prices.get(row_model or "")
    if not p:
        return NAN
    return ((tin or 0) * p.get("input", 0) + (tout or 0) * p.get("output", 0)) / 1e6


def cost_latency(conn, cfg) -> tuple[list[dict], list[dict]]:
    prices = cfg.prices
    rows = conn.execute("SELECT * FROM call_log").fetchall()
    if not rows:
        raise Skipped("call_log is empty")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["provider"], r["model"], r["purpose"])].append(r)
    by_purpose = []
    for (prov, model, purpose), rs in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        lat = [r["latency_ms"] for r in rs if r["latency_ms"] is not None]
        costs = [_cost(model, r["input_tokens"], r["output_tokens"], r["cost_usd"], prices) for r in rs]
        by_purpose.append({"provider": prov, "model": model, "purpose": purpose, "calls": len(rs),
                           "errors": sum(1 for r in rs if r["error"]),
                           "input_tokens": sum(r["input_tokens"] or 0 for r in rs),
                           "output_tokens": sum(r["output_tokens"] or 0 for r in rs),
                           "latency_mean_ms": statistics.mean(lat) if lat else NAN,
                           "latency_median_ms": statistics.median(lat) if lat else NAN,
                           "cost_usd": NAN if any(c != c for c in costs) else sum(costs)})
    per_run = []
    for run in conn.execute("SELECT * FROM extraction_run ORDER BY extraction_run_id"):
        calls = conn.execute("""SELECT DISTINCT c.* FROM call_log c JOIN model_response m ON m.call_id = c.call_id
                                WHERE m.extraction_run_id = ?""", (run["extraction_run_id"],)).fetchall()
        n_art = len(repo.get_articles(conn, run["corpus_id"], selected_only=True)) or 1
        costs = [_cost(c["model"], c["input_tokens"], c["output_tokens"], c["cost_usd"], prices) for c in calls]
        cost = NAN if any(c != c for c in costs) else sum(costs)
        lat = sum(c["latency_ms"] or 0 for c in calls)
        per_run.append({"source": "extraction", "id": run["extraction_run_id"], "corpus_id": run["corpus_id"],
                        "model": run["model"], "articles": n_art, "calls": len(calls),
                        "input_tokens": sum(c["input_tokens"] or 0 for c in calls), "cost_usd": cost,
                        "cost_per_article_usd": cost / n_art, "latency_s": lat / 1000,
                        "latency_per_article_s": lat / 1000 / n_art})
    for b in conn.execute("SELECT b.*, c.model AS call_model, c.input_tokens, c.output_tokens, c.cost_usd, c.latency_ms "
                          "FROM baseline_assessment b LEFT JOIN call_log c USING (call_id)"):
        n_art = len(repo.get_articles(conn, b["corpus_id"], selected_only=True)) or 1
        cost = _cost(b["call_model"], b["input_tokens"], b["output_tokens"], b["cost_usd"], prices)
        per_run.append({"source": "baseline", "id": b["baseline_id"], "corpus_id": b["corpus_id"], "model": b["model"],
                        "articles": n_art, "calls": 1, "input_tokens": b["input_tokens"] or 0, "cost_usd": cost,
                        "cost_per_article_usd": cost / n_art, "latency_s": (b["latency_ms"] or 0) / 1000,
                        "latency_per_article_s": (b["latency_ms"] or 0) / 1000 / n_art})
    return by_purpose, per_run


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

METRICS = {
    "search_reproducibility": (search_reproducibility, ["search_reproducibility"]),
    "extraction_quality": (extraction_quality, ["extraction_quality", "extraction_quality_by_criterion"]),
    "hallucination": (hallucination, ["hallucination"]),
    "extraction_stability": (extraction_stability, ["extraction_stability"]),
    "calibration": (calibration, ["calibration", "calibration_reliability_bins"]),
    "scoring_robustness": (scoring_robustness, ["scoring_robustness_variants", "scoring_robustness_ranking"]),
    "baseline_comparison": (baseline_comparison, ["baseline_comparison_by_criterion", "baseline_comparison"]),
    "report_reliability": (report_reliability, ["report_reliability_reports", "report_reliability"]),
    "cost_latency": (cost_latency, ["cost_latency_by_purpose", "cost_latency_per_run"]),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ["(empty)"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and v != v else v) for k, v in r.items()})


def run_metrics(conn, cfg: AppConfig, names: list[str] | None = None, out_dir: str | Path = "results") -> list[tuple[str, str]]:
    """Compute the given metrics (default: all) and write CSVs. Returns (metric, outcome) pairs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    results = []
    for name in names or list(METRICS):
        fn, files = METRICS[name]
        try:
            res = fn(conn, cfg)
        except Skipped as exc:
            results.append((name, f"skipped: {exc}"))
            continue
        tables = res if isinstance(res, tuple) else (res,)
        written = []
        for file, rows in zip(files, tables):
            path = out / f"eval_{file}.csv"
            write_csv(path, rows)
            written.append(f"{path} ({len(rows)} rows)")
        results.append((name, "; ".join(written) + f" [{stamp}]"))
    return results
