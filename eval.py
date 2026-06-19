"""Ranking evaluation. OFFLINE-ONLY: reads hn_rewrite.db exclusively.

Compares 3 score formulas via 5-fold stratified CV:
  current:     P(up) + 0.5 * P(neutral)   [production]
  up_only:     P(up)
  hn_baseline: raw HN points (no SVM)

Personalization features are computed per-fold (LOOCV self-exclusion)
to avoid train-test leakage. MMR and raw (pre-MMR) metrics both reported.

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
from sklearn.preprocessing import StandardScaler

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


def _compute_metrics(
    rank_map: dict[int, int],
    test_stories: list[Story],
    test_actions: np.ndarray,
    test_rel: np.ndarray,
    all_test_rels: list[float],
) -> dict:
    rel_by_pos = {}
    for i, ts in enumerate(test_stories):
        if ts.id in rank_map:
            rel_by_pos[rank_map[ts.id]] = test_rel[i]

    mrr_vals = [
        1.0 / (rank_map[ts.id] + 1)
        for i, ts in enumerate(test_stories)
        if test_actions[i] == 2 and ts.id in rank_map
    ]

    def _ndcg(
        rel_by_pos: dict[int, float], all_test_rels: list[float], k: int
    ) -> float:
        dcg = sum(r / math.log2(p + 2) for p, r in rel_by_pos.items() if p < k)
        ideal = sorted(all_test_rels, reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    return {
        "ndcg_at_5": _ndcg(rel_by_pos, all_test_rels, 5),
        "ndcg_at_10": _ndcg(rel_by_pos, all_test_rels, 10),
        "ndcg_at_20": _ndcg(rel_by_pos, all_test_rels, 20),
        "ndcg_at_40": _ndcg(rel_by_pos, all_test_rels, 40),
        "hit_at_40": len(rel_by_pos) / max(len(test_stories), 1),
        "mrr": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
    }


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

    rel_map = {0: 0.0, 1: 0.5, 2: 1.0}
    test_rel = np.array([rel_map[int(a)] for a in test_actions])
    all_test_rels = test_rel.tolist()

    # MMR-based ranking (production path)
    top40 = mmr_filter(ranked, emb_map, threshold=mmr_threshold, limit=mmr_limit)
    mmr_rank_map = {rs.story.id: pos for pos, rs in enumerate(top40)}
    mmr_metrics = _compute_metrics(
        mmr_rank_map, test_stories, test_actions, test_rel, all_test_rels
    )

    # Raw ranking (no MMR, diagnostic)
    raw_ranked = ranked[:mmr_limit]
    raw_rank_map = {rs.story.id: pos for pos, rs in enumerate(raw_ranked)}
    raw_metrics = _compute_metrics(
        raw_rank_map, test_stories, test_actions, test_rel, all_test_rels
    )

    # Calculate median rank of upvoted test stories in overall ranked candidates
    test_upvote_ids = {s.id for i, s in enumerate(test_stories) if test_actions[i] == 2}
    upvote_ranks = [
        pos for pos, idx in enumerate(order)
        if candidates[idx].id in test_upvote_ids
    ]
    median_rank = float(np.median(upvote_ranks)) if upvote_ranks else 0.0

    mmr_metrics["median_rank"] = median_rank
    raw_metrics["median_rank"] = median_rank

    return {"mmr": mmr_metrics, "raw": raw_metrics}


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
    cand_comment_counts = np.array([s.comment_count or 0 for s in candidates])
    cand_text_lengths = np.array([len(s.text_content) for s in candidates])
    cand_ages_arr = np.array([now - max(s.time, 1) for s in candidates])
    cand_scores_arr = np.array([s.score for s in candidates])
    cand_quality_arr = cand_scores_arr / (np.maximum(cand_ages_arr / 3600.0, 0) + 1)

    # Feedback metadata (age = vote_time - story.time)
    fb_emb = cand_emb[fb_to_cand[valid]]
    fb_scores_arr = np.array([s.score for s in fb_stories])[valid]
    fb_ages_arr = np.array(
        [float(vt) - max(s.time, 1) for vt, s in zip(fb_vote_times, fb_stories)]
    )[valid]
    fb_comment_counts_arr = np.array([s.comment_count or 0 for s in fb_stories])[valid]
    fb_text_lengths_arr = np.array([len(s.text_content) for s in fb_stories])[valid]
    fb_quality_arr = fb_scores_arr / (np.maximum(fb_ages_arr / 3600.0, 0) + 1)

    cand_scores_array = np.array([s.score for s in candidates], dtype=np.float64)
    y = fb_labels[valid]

    formulas = ["current", "up_only", "hn_baseline"]
    results: dict[str, list[dict]] = {f: [] for f in formulas}

    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(
            np.zeros((len(y), 1)), y
        )
    )

    for fold_idx, (train_pos, test_pos) in enumerate(folds):
        # Slice training data for this fold
        fb_train_emb = fb_emb[train_pos]
        y_train = y[train_pos]

        fb_train_scores = fb_scores_arr[train_pos]
        fb_train_ages = fb_ages_arr[train_pos]
        fb_train_comments = fb_comment_counts_arr[train_pos]
        fb_train_textlens = fb_text_lengths_arr[train_pos]
        fb_train_quality = fb_quality_arr[train_pos]

        # Per-fold personalization with LOOCV self-exclusion
        up_mask = y_train == 2
        down_mask = y_train == 0
        fb_up_train = fb_train_emb[up_mask]
        fb_down_train = fb_train_emb[down_mask]
        n_up = up_mask.sum()
        n_down = down_mask.sum()

        mean_up = fb_up_train.mean(axis=0) if n_up else np.zeros(384, dtype=np.float32)
        mean_down = (
            fb_down_train.mean(axis=0) if n_down else np.zeros(384, dtype=np.float32)
        )

        # Candidate features (from train centroids — no self issue)
        cand_sim_up = cand_emb @ mean_up
        cand_sim_down = cand_emb @ mean_down
        cand_closest_up = (
            np.max(cand_emb @ fb_up_train.T, axis=1)
            if n_up
            else np.zeros(len(candidates))
        )
        cand_closest_down = (
            np.max(cand_emb @ fb_down_train.T, axis=1)
            if n_down
            else np.zeros(len(candidates))
        )

        # Train features: exclude self-contribution (LOOCV)
        fb_sim_up = fb_train_emb @ mean_up
        fb_sim_down = fb_train_emb @ mean_down

        if n_up > 0:
            if n_up > 1:
                fb_sim_up[up_mask] = (n_up * fb_sim_up[up_mask] - 1.0) / (n_up - 1)
            else:
                fb_sim_up[up_mask] = 0.0

            sim_up_mat = fb_train_emb @ fb_up_train.T
            if n_up > 1:
                train_up_positions = np.where(up_mask)[0]
                for i, tp in enumerate(train_up_positions):
                    sim_up_mat[tp, i] = -1.0
            fb_closest_up = np.max(sim_up_mat, axis=1)
        else:
            fb_closest_up = np.zeros(len(fb_train_emb))

        if n_down > 0:
            if n_down > 1:
                fb_sim_down[down_mask] = (n_down * fb_sim_down[down_mask] - 1.0) / (
                    n_down - 1
                )
            else:
                fb_sim_down[down_mask] = 0.0

            sim_down_mat = fb_train_emb @ fb_down_train.T
            if n_down > 1:
                train_down_positions = np.where(down_mask)[0]
                for i, tp in enumerate(train_down_positions):
                    sim_down_mat[tp, i] = -1.0
            fb_closest_down = np.max(sim_down_mat, axis=1)
        else:
            fb_closest_down = np.zeros(len(fb_train_emb))

        # Build features for this fold
        X_train = _augment_features(
            fb_train_emb,
            fb_train_scores,
            fb_train_ages,
            comment_counts=fb_train_comments,
            text_lengths=fb_train_textlens,
            hn_quality=fb_train_quality,
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

        counts = Counter(y_train)
        weights = np.array(
            [len(y_train) / (3 * counts[c]) for c in y_train], dtype=np.float64
        )

        emb_dim = cand_emb.shape[1]
        scaler = StandardScaler()
        X_train_meta_scaled = scaler.fit_transform(X_train[:, emb_dim:])
        X_cand_meta_scaled = scaler.transform(X_cand[:, emb_dim:])

        X_train_scaled = np.hstack([X_train[:, :emb_dim], X_train_meta_scaled])
        X_cand_scaled = np.hstack([X_cand[:, :emb_dim], X_cand_meta_scaled])

        svm = SVC(
            C=config.model.svm_c,
            kernel=config.model.svm_kernel,
            gamma=config.model.svm_gamma,
            random_state=0,
            decision_function_shape="ovr",
            probability=True,
        )
        svm.fit(X_train_scaled, y_train, sample_weight=weights)
        probs = svm.predict_proba(X_cand_scaled)

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
    metric_keys = (
        "ndcg_at_5",
        "ndcg_at_10",
        "ndcg_at_20",
        "ndcg_at_40",
        "hit_at_40",
        "mrr",
        "median_rank",
    )

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
                    "mmr": {
                        k: float(np.mean([r["mmr"][k] for r in rs]))
                        for k in metric_keys
                    },
                    "raw": {
                        k: float(np.mean([r["raw"][k] for r in rs]))
                        for k in metric_keys
                    },
                },
                "std": {
                    "mmr": {
                        k: float(np.std([r["mmr"][k] for r in rs])) for k in metric_keys
                    },
                    "raw": {
                        k: float(np.std([r["raw"][k] for r in rs])) for k in metric_keys
                    },
                },
                "per_fold": rs,
            }
            for f, rs in results.items()
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWritten {REPORT_PATH}")

    for metric in ("ndcg_at_10", "hit_at_40", "mrr", "median_rank"):
        print(f"\n{metric} by formula (mean ± std):")
        for f, data in report["formulas"].items():  # type: ignore[union-attr]
            for variant in ("mmr", "raw"):
                m, s = data["mean"][variant][metric], data["std"][variant][metric]  # type: ignore
                print(f"  {f:12s} {variant:4s} {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
