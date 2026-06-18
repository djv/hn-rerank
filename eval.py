"""Ranking evaluation. OFFLINE-ONLY: reads hn_rewrite.db exclusively.

Compares 4 score formulas via 5-fold stratified CV:
  current:     P(up) + 0.5 * P(neutral)   [production]
  diff:        P(up) - P(down)
  up_only:     P(up)
  hn_baseline: raw HN points (no SVM)

Writes eval_report.json (committed to git for tracking).
"""

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from database import Database, Story
from pipeline import (
    Config,
    RankedStory,
    _augment_features,
    mmr_filter,
)

MODEL_VERSION = "all-MiniLM-L6-v2|mean|norm|256"
REPORT_PATH = Path(__file__).parent / "eval_report.json"


def _db_sha256(db_path: str) -> str:
    return hashlib.sha256(Path(db_path).read_bytes()).hexdigest()[:16]


def _load_candidates(db: Database) -> tuple[list[Story], np.ndarray]:
    """Read all non-negative-cached stories + their embeddings."""
    cursor = db.conn.execute(
        "SELECT id, title, url, score, time, text_content, source, "
        "       comment_count, discussion_url "
        "FROM stories WHERE text_content != ''"
    )
    stories = [
        Story(
            id=row[0],
            title=row[1],
            url=row[2],
            score=row[3],
            time=row[4],
            text_content=row[5],
            source=row[6],
            comment_count=row[7],
            discussion_url=row[8],
        )
        for row in cursor.fetchall()
    ]
    cached = db.get_embeddings_batch([s.id for s in stories], MODEL_VERSION)
    embeddings = np.array(
        [cached.get(s.id, np.zeros(384, dtype=np.float32)) for s in stories],
        dtype=np.float32,
    )
    return stories, embeddings


def _ndcg(rel_by_pos: dict[int, float], all_rels: list[float], k: int) -> float:
    """Graded NDCG@k: sum_{pos<k} rel(pos)/log2(pos+2) / IDCG."""
    dcg = sum(r / math.log2(p + 2) for p, r in rel_by_pos.items() if p < k)
    ideal = sorted(all_rels, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _evaluate_fold(
    probs: np.ndarray,
    candidates: list[Story],
    cand_emb: np.ndarray,
    test_stories: list[Story],
    test_actions: np.ndarray,
    cand_scores: np.ndarray,
    formula: str,
    mmr_threshold: float = 0.85,
    mmr_limit: int = 40,
) -> dict:
    if formula == "current":
        scores = probs[:, 2] + 0.5 * probs[:, 1]
    elif formula == "diff":
        scores = probs[:, 2] - probs[:, 0]
    elif formula == "up_only":
        scores = probs[:, 2]
    elif formula == "hn_baseline":
        scores = cand_scores.astype(np.float32)
    else:
        raise ValueError(f"Unknown formula: {formula}")

    order = np.argsort(-scores)
    ranked = [
        RankedStory(story=candidates[i], score=float(scores[i]), best_match_title="")
        for i in order
    ]
    emb_map = {candidates[i].id: cand_emb[i] for i in range(len(candidates))}
    top40 = mmr_filter(ranked, emb_map, threshold=mmr_threshold, limit=mmr_limit)
    top40_rank = {rs.story.id: pos for pos, rs in enumerate(top40)}

    rel_map = {0: 0.0, 1: 0.5, 2: 1.0}
    test_rel = np.array([rel_map[int(a)] for a in test_actions])

    rel_by_pos = {}
    all_rels = []
    for i, ts in enumerate(test_stories):
        if ts.id in top40_rank:
            pos = top40_rank[ts.id]
            rel_by_pos[pos] = test_rel[i]
            all_rels.append(test_rel[i])

    # MRR: reciprocal rank of first upvote in top-40
    mrr_vals = [
        1.0 / (top40_rank[ts.id] + 1)
        for i, ts in enumerate(test_stories)
        if test_actions[i] == 2 and ts.id in top40_rank
    ]

    return {
        "ndcg_at_5": _ndcg(rel_by_pos, all_rels, 5),
        "ndcg_at_10": _ndcg(rel_by_pos, all_rels, 10),
        "ndcg_at_20": _ndcg(rel_by_pos, all_rels, 20),
        "ndcg_at_40": _ndcg(rel_by_pos, all_rels, 40),
        "hit_at_40": len(all_rels) / max(len(test_stories), 1),
        "mrr": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
    }


def main() -> None:
    config = Config.load()
    db = Database(config.db_path)

    # Feedback
    fb_stories, fb_labels, fb_vote_times = db.get_feedback_for_training()
    fb_labels = np.array(fb_labels, dtype=int)
    fb_vote_times = np.array(fb_vote_times, dtype=np.float64)
    print(f"Feedback: {len(fb_stories)} rows ({Counter(fb_labels)})")

    # Candidates
    candidates, cand_emb = _load_candidates(db)
    print(f"Candidates: {len(candidates)}")

    # Map feedback stories → candidate indices
    cand_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
    fb_to_cand = np.array([cand_id_to_idx.get(s.id, -1) for s in fb_stories], dtype=int)
    valid = fb_to_cand >= 0
    if not valid.all():
        print(
            f"Warning: {(~valid).sum()} feedback stories missing from candidates; excluded."
        )

    # Candidate features (age = now - story.time)
    now = time.time()
    cand_ages = [now - max(s.time, 1) for s in candidates]
    cand_scores_list = [s.score for s in candidates]
    cand_comment_counts = np.array([s.comment_count or 0 for s in candidates])
    cand_text_lengths = np.array([len(s.text_content) for s in candidates])
    cand_ages_arr = np.array(cand_ages)
    cand_scores_arr = np.array(cand_scores_list)
    cand_quality_arr = cand_scores_arr / (np.maximum(cand_ages_arr / 3600.0, 0) + 1)

    # Feedback features (age = vote_time - story.time)
    fb_emb = cand_emb[fb_to_cand[valid]]
    fb_scores_arr = np.array([s.score for s in fb_stories])[valid]
    fb_ages_arr = np.array(
        [float(vt) - max(s.time, 1) for vt, s in zip(fb_vote_times, fb_stories)]
    )[valid]
    fb_comment_counts_arr = np.array([s.comment_count or 0 for s in fb_stories])[valid]
    fb_text_lengths_arr = np.array([len(s.text_content) for s in fb_stories])[valid]
    fb_quality_arr = fb_scores_arr / (np.maximum(fb_ages_arr / 3600.0, 0) + 1)

    # Personalization features: computed once from ALL valid feedback
    # (matches production — user profile from all known feedback)
    y = fb_labels[valid]
    up_mask = y == 2
    down_mask = y == 0
    fb_up_embs = fb_emb[up_mask]
    fb_down_embs = fb_emb[down_mask]

    mean_up = (
        fb_up_embs.mean(axis=0) if up_mask.any() else np.zeros(384, dtype=np.float32)
    )
    mean_down = (
        fb_down_embs.mean(axis=0)
        if down_mask.any()
        else np.zeros(384, dtype=np.float32)
    )

    fb_sim_up = fb_emb @ mean_up
    fb_sim_down = fb_emb @ mean_down
    fb_closest_up = (
        np.max(fb_emb @ fb_up_embs.T, axis=1)
        if up_mask.any()
        else np.zeros(len(fb_emb))
    )
    fb_closest_down = (
        np.max(fb_emb @ fb_down_embs.T, axis=1)
        if down_mask.any()
        else np.zeros(len(fb_emb))
    )

    cand_sim_up = cand_emb @ mean_up
    cand_sim_down = cand_emb @ mean_down
    cand_closest_up = (
        np.max(cand_emb @ fb_up_embs.T, axis=1)
        if up_mask.any()
        else np.zeros(len(candidates))
    )
    cand_closest_down = (
        np.max(cand_emb @ fb_down_embs.T, axis=1)
        if down_mask.any()
        else np.zeros(len(candidates))
    )

    # Build full feature arrays (once, outside fold loop)
    X_fb = _augment_features(
        fb_emb,
        fb_scores_arr,
        fb_ages_arr,
        comment_counts=fb_comment_counts_arr,
        text_lengths=fb_text_lengths_arr,
        hn_quality=fb_quality_arr,
        sim_to_upvoted=fb_sim_up,
        sim_to_downvoted=fb_sim_down,
        closest_upvoted=fb_closest_up,
        closest_downvoted=fb_closest_down,
    )
    X_cand = _augment_features(
        cand_emb,
        cand_scores_arr,
        cand_ages_arr,
        comment_counts=cand_comment_counts,
        text_lengths=cand_text_lengths,
        hn_quality=cand_quality_arr,
        sim_to_upvoted=cand_sim_up,
        sim_to_downvoted=cand_sim_down,
        closest_upvoted=cand_closest_up,
        closest_downvoted=cand_closest_down,
    )

    cand_scores_array = np.array([s.score for s in candidates], dtype=np.float64)

    formulas = ["current", "diff", "up_only", "hn_baseline"]
    results: dict[str, list[dict]] = {f: [] for f in formulas}

    # 5-fold CV
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X_fb, y)
    )

    for fold_idx, (train_pos, test_pos) in enumerate(folds):
        X_train = X_fb[train_pos]
        y_train = y[train_pos]

        counts = Counter(y_train)
        weights = np.array(
            [len(y_train) / (3 * counts[c]) for c in y_train], dtype=np.float64
        )

        svm = SVC(
            C=config.model.svm_c,
            kernel=config.model.svm_kernel,
            gamma=config.model.svm_gamma,
            random_state=0,
            decision_function_shape="ovr",
        )
        svm.fit(X_train, y_train, sample_weight=weights)
        n_train = len(X_train)
        calibrated = CalibratedClassifierCV(
            svm, cv=[(list(range(n_train)), list(range(n_train)))], method="sigmoid"
        )
        calibrated.fit(X_train, y_train, sample_weight=weights)
        probs = calibrated.predict_proba(X_cand)

        # Test fold: map test positions back to stories
        test_stories = [
            fb_stories[valid_idx] for valid_idx in np.where(valid)[0][test_pos]
        ]
        test_actions = y[test_pos]

        for formula in formulas:
            results[formula].append(
                _evaluate_fold(
                    probs,
                    candidates,
                    cand_emb,
                    test_stories,
                    test_actions,
                    cand_scores_array,
                    formula,
                )
            )

        print(f"Fold {fold_idx + 1}/5 done")

    # Aggregate
    report = {
        "config": {
            "split": "5-fold-stratified",
            "random_state": 0,
            "n_feedback": int(len(fb_labels)),
            "n_candidates": int(len(candidates)),
            "n_folds": 5,
            "mmr_threshold": 0.85,
            "mmr_limit": 40,
            "relevance_grade": "up=1, neutral=0.5, down=0",
            "db_sha256": _db_sha256(config.db_path),
        },
        "formulas": {
            f: {
                "mean": {
                    k: float(np.mean([r[k] for r in rs]))
                    for k in (
                        "ndcg_at_5",
                        "ndcg_at_10",
                        "ndcg_at_20",
                        "ndcg_at_40",
                        "hit_at_40",
                        "mrr",
                    )
                },
                "std": {
                    k: float(np.std([r[k] for r in rs]))
                    for k in (
                        "ndcg_at_5",
                        "ndcg_at_10",
                        "ndcg_at_20",
                        "ndcg_at_40",
                        "hit_at_40",
                        "mrr",
                    )
                },
                "per_fold": rs,
            }
            for f, rs in results.items()
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWritten {REPORT_PATH}")

    for metric in ("ndcg_at_10", "hit_at_40", "mrr"):
        print(f"\n{metric} by formula (mean ± std):")
        for f, data in report["formulas"].items():  # type: ignore[union-attr]
            m, s = data["mean"][metric], data["std"][metric]  # type: ignore
            print(f"  {f:15s}  {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
