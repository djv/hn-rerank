"""Feature ablation: compare baseline vs. extra meta-features.

Each variant adds extra columns to the 392-dim feature vector.
Runs 5-fold CV and prints comparison. No file writes.
"""

import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from database import Database, Story
from eval import _load_candidates, _evaluate_fold
from pipeline import Config, _augment_features

LOG_POINTS_SCALE = 2.0
LOG_COMMENTS_SCALE = 1.0
LOG_TEXTLEN_SCALE = 2.0
LOG_QUALITY_SCALE = 8.0


def _title_lexical(stories: list[Story]) -> np.ndarray:
    n = len(stories)
    out = np.zeros((n, 4), dtype=np.float32)
    for i, s in enumerate(stories):
        t = s.title or ""
        out[i, 0] = 1.0 if "?" in t else 0.0  # has_question
        out[i, 1] = np.log1p(len(t)) / 6.0  # title_len (normalized)
        out[i, 2] = 1.0 if not s.url else 0.0  # is_self_post
        caps = sum(1 for c in t if c.isupper())
        out[i, 3] = caps / max(len(t), 1)  # caps_ratio
    return out


def _domain_onehot(stories: list[Story]) -> np.ndarray:
    """One-hot encode top-level domain, + 'other' bucket."""
    domains: list[str] = []
    for s in stories:
        url = s.url or ""
        if not url:
            domains.append("self")
        else:
            try:
                parts = urlparse(url).netloc.split(".")
                if len(parts) >= 2:
                    domains.append(parts[-2].lower())
                else:
                    domains.append("unknown")
            except Exception:
                domains.append("unknown")
    # Top 15 most common domains
    counts = Counter(domains)
    top = [d for d, _ in counts.most_common(15)]
    domain_set = set(top)
    n = len(stories)
    out = np.zeros((n, len(top) + 1), dtype=np.float32)
    for i, d in enumerate(domains):
        if d in domain_set:
            out[i, top.index(d)] = 1.0
        else:
            out[i, -1] = 1.0  # other bucket
    return out


def _velocity_features(
    scores: np.ndarray, ages_h: np.ndarray, comment_counts: np.ndarray
) -> np.ndarray:
    n = len(scores)
    out = np.zeros((n, 2), dtype=np.float32)
    safe_h = np.maximum(ages_h, 0.1)
    vel = scores / safe_h
    out[:, 0] = np.clip(np.log1p(np.maximum(vel, 0)), 0, 5.0) / 5.0
    cv = comment_counts / safe_h
    out[:, 1] = np.clip(np.log1p(np.maximum(cv, 0)), 0, 5.0) / 5.0
    return out


def _comment_score_ratio(comment_counts: np.ndarray, scores: np.ndarray) -> np.ndarray:
    ratio = comment_counts / np.maximum(scores, 1)
    return np.clip(np.log1p(ratio), 0, 3.0).reshape(-1, 1) / 3.0


def _main():
    config = Config.load()
    db = Database(config.db_path)
    print(f"DB: {config.db_path}")

    fb_stories, fb_labels, fb_vote_times = db.get_feedback_for_training()
    fb_labels = np.array(fb_labels, dtype=int)
    fb_vote_times = np.array(fb_vote_times, dtype=np.float64)
    print(f"Feedback: {len(fb_stories)} rows ({Counter(fb_labels)})")

    candidates, cand_emb = _load_candidates(db)
    print(f"Candidates: {len(candidates)}")

    cand_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
    fb_to_cand = np.array([cand_id_to_idx.get(s.id, -1) for s in fb_stories], dtype=int)
    valid = fb_to_cand >= 0
    if not valid.all():
        print(f"Warning: {(~valid).sum()} feedback stories missing")

    now = time.time()
    cand_comment_counts = np.array([s.comment_count or 0 for s in candidates])
    cand_text_lengths = np.array([len(s.text_content) for s in candidates])
    cand_ages_arr = np.array([now - max(s.time, 1) for s in candidates])
    cand_scores_arr = np.array([s.score for s in candidates])
    cand_quality_arr = cand_scores_arr / (np.maximum(cand_ages_arr / 3600.0, 0) + 1)

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

    # Precompute fold-independent extra features for all candidates + feedback
    cand_title_lex = _title_lexical(candidates)
    cand_domain_oh = _domain_onehot(candidates)
    cand_v = _velocity_features(
        cand_scores_arr, cand_ages_arr / 3600, cand_comment_counts
    )
    cand_csr = _comment_score_ratio(cand_comment_counts, cand_scores_arr)

    fb_title_lex = _title_lexical(fb_stories)
    fb_domain_oh = _domain_onehot(fb_stories)
    fb_v = _velocity_features(fb_scores_arr, fb_ages_arr / 3600, fb_comment_counts_arr)
    fb_csr = _comment_score_ratio(fb_comment_counts_arr, fb_scores_arr)

    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(
            np.zeros((len(y), 1)), y
        )
    )

    # Define variants: (name, build_extra_fn(fb_idx, cand_mask)->(X_train_extra, X_cand_extra))
    def baseline_extra(train_pos, cand_mask):
        return np.zeros((len(train_pos), 0), dtype=np.float32), np.zeros(
            (int(cand_mask.sum()), 0), dtype=np.float32
        )

    def title_lex_extra(train_pos, cand_mask):
        return fb_title_lex[train_pos], cand_title_lex[cand_mask]

    def domain_extra(train_pos, cand_mask):
        return fb_domain_oh[train_pos], cand_domain_oh[cand_mask]

    def vel_extra(train_pos, cand_mask):
        return fb_v[train_pos], cand_v[cand_mask]

    def csr_extra(train_pos, cand_mask):
        return fb_csr[train_pos], cand_csr[cand_mask]

    def time_decayed_extra_fn(now):
        def _fn(train_pos, cand_mask):
            # EWMA-weighted upvote centroid
            train_up_pos = [p for p in train_pos if y[p] == 2]
            if not train_up_pos:
                return np.zeros((len(train_pos), 0), dtype=np.float32), np.zeros(
                    (int(cand_mask.sum()), 0), dtype=np.float32
                )
            fb_up_times = np.array([float(fb_vote_times[idx]) for idx in train_up_pos])
            weights = np.exp(-(now - fb_up_times) / (30 * 24 * 3600))
            fb_up_emb = np.array(
                [fb_emb[list(train_pos).index(p)] for p in train_up_pos]
            )
            weighted_mean_up = np.average(fb_up_emb, axis=0, weights=weights)
            weighted_mean_up /= max(np.linalg.norm(weighted_mean_up), 1e-8)

            # Similarity to weighted centroid
            fb_sim = fb_emb[train_pos] @ weighted_mean_up
            cand_sim_cand = cand_emb[cand_mask] @ weighted_mean_up

            fb_extra = (np.clip(fb_sim, -1, 1) + 1).reshape(-1, 1) / 2
            cand_extra = (np.clip(cand_sim_cand, -1, 1) + 1).reshape(-1, 1) / 2
            return fb_extra.astype(np.float32), cand_extra.astype(np.float32)

        return _fn

    def all_extra_fn(now):
        tde_fn = time_decayed_extra_fn(now)

        def _fn(train_pos, cand_mask):
            fb_tde, cand_tde = tde_fn(train_pos, cand_mask)
            fb_list = [
                fb_title_lex[train_pos],
                fb_domain_oh[train_pos],
                fb_v[train_pos],
                fb_csr[train_pos],
            ]
            cand_list = [
                cand_title_lex[cand_mask],
                cand_domain_oh[cand_mask],
                cand_v[cand_mask],
                cand_csr[cand_mask],
            ]
            if fb_tde.shape[1] > 0:
                fb_list.append(fb_tde)
                cand_list.append(cand_tde)
            return np.hstack(fb_list).astype(np.float32), np.hstack(cand_list).astype(
                np.float32
            )

        return _fn

    variants = [
        ("baseline", baseline_extra),
        ("+title_lexical", title_lex_extra),
        ("+domain_onehot", domain_extra),
        ("+velocity", vel_extra),
        ("+comment_score_ratio", csr_extra),
        ("+time_decayed_profile", time_decayed_extra_fn(now)),
        (
            "+all_non_personal",
            lambda tp, cm: (
                np.hstack(
                    [fb_title_lex[tp], fb_domain_oh[tp], fb_v[tp], fb_csr[tp]]
                ).astype(np.float32),
                np.hstack(
                    [cand_title_lex[cm], cand_domain_oh[cm], cand_v[cm], cand_csr[cm]]
                ).astype(np.float32),
            ),
        ),
        ("+all_extra", all_extra_fn(now)),
    ]

    metric_keys = ("ndcg_at_40", "map", "brier_up", "median_rank")
    all_results: dict[str, list[dict]] = {}

    for var_name, build_extra in variants:
        print(f"\n{'=' * 60}")
        print(f"Variant: {var_name}")
        print("=" * 60)
        fold_results = []

        for fold_idx, (train_pos, test_pos) in enumerate(folds):
            fb_train_emb = fb_emb[train_pos]
            y_train = y[train_pos]
            fb_train_scores = fb_scores_arr[train_pos]
            fb_train_ages = fb_ages_arr[train_pos]
            fb_train_comments = fb_comment_counts_arr[train_pos]
            fb_train_textlens = fb_text_lengths_arr[train_pos]
            fb_train_quality = fb_quality_arr[train_pos]

            train_ids = {fb_stories[idx].id for idx in np.where(valid)[0][train_pos]}
            cand_mask = np.array([s.id not in train_ids for s in candidates])
            fold_candidates = [s for idx, s in enumerate(candidates) if cand_mask[idx]]
            fold_cand_emb = cand_emb[cand_mask]
            fold_cand_scores = cand_scores_arr[cand_mask]
            fold_cand_ages = cand_ages_arr[cand_mask]
            fold_cand_comments = cand_comment_counts[cand_mask]
            fold_cand_textlens = cand_text_lengths[cand_mask]
            fold_cand_quality = cand_quality_arr[cand_mask]
            fold_cand_scores_array = cand_scores_array[cand_mask]

            up_mask = y_train == 2
            down_mask = y_train == 0
            fb_up_train = fb_train_emb[up_mask]
            fb_down_train = fb_train_emb[down_mask]
            n_up = up_mask.sum()
            n_down = down_mask.sum()

            mean_up = (
                fb_up_train.mean(axis=0) if n_up else np.zeros(384, dtype=np.float32)
            )
            mean_down = (
                fb_down_train.mean(axis=0)
                if n_down
                else np.zeros(384, dtype=np.float32)
            )

            cand_sim_up = fold_cand_emb @ mean_up
            cand_sim_down = fold_cand_emb @ mean_down
            cand_closest_up = (
                np.max(fold_cand_emb @ fb_up_train.T, axis=1)
                if n_up
                else np.zeros(len(fold_candidates))
            )
            cand_closest_down = (
                np.max(fold_cand_emb @ fb_down_train.T, axis=1)
                if n_down
                else np.zeros(len(fold_candidates))
            )

            fb_sim_up = fb_train_emb @ mean_up
            fb_sim_down = fb_train_emb @ mean_down

            if n_up > 0:
                if n_up > 1:
                    fb_sim_up[up_mask] = (n_up * fb_sim_up[up_mask] - 1.0) / (n_up - 1)
                else:
                    fb_sim_up[up_mask] = 0.0
                sim_up_mat = fb_train_emb @ fb_up_train.T
                if n_up > 1:
                    for i, tp in enumerate(np.where(up_mask)[0]):
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
                    for i, tp in enumerate(np.where(down_mask)[0]):
                        sim_down_mat[tp, i] = -1.0
                fb_closest_down = np.max(sim_down_mat, axis=1)
            else:
                fb_closest_down = np.zeros(len(fb_train_emb))

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
                fold_cand_emb,
                fold_cand_scores,
                fold_cand_ages,
                comment_counts=fold_cand_comments,
                text_lengths=fold_cand_textlens,
                hn_quality=fold_cand_quality,
                sim_to_upvoted=cand_sim_up,
                sim_to_downvoted=cand_sim_down,
                closest_upvoted=cand_closest_up,
                closest_downvoted=cand_closest_down,
            )

            # Add extra features
            X_train_extra, X_cand_extra = build_extra(train_pos, cand_mask)
            if X_train_extra.shape[1] > 0:
                X_train = np.hstack([X_train, X_train_extra])
                X_cand = np.hstack([X_cand, X_cand_extra])

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

            test_stories = [
                fb_stories[valid_idx] for valid_idx in np.where(valid)[0][test_pos]
            ]
            test_actions = y[test_pos]

            fold_results.append(
                _evaluate_fold(
                    probs,
                    fold_candidates,
                    fold_cand_emb,
                    test_stories,
                    test_actions,
                    fold_cand_scores_array,
                    "up_only",
                    mmr_threshold=config.model.diversity_threshold,
                )
            )
            print(f"  Fold {fold_idx + 1}/5 done")

        all_results[var_name] = fold_results

    # Print comparison table (mmr metrics)
    print(f"\n{'=' * 80}")
    print(
        f"{'Variant':30s} {'NDCG@40':>10s} {'MAP':>10s} {'Brier':>10s} {'MedRank':>10s}"
    )
    print("-" * 70)
    for var_name in [v[0] for v in variants]:
        fr = all_results[var_name]
        means = {k: float(np.mean([r["mmr"][k] for r in fr])) for k in metric_keys}
        print(
            f"{var_name:30s} {means['ndcg_at_40']:10.4f} {means['map']:10.4f} "
            f"{means['brier_up']:10.4f} {means['median_rank']:8.1f}"
        )
    print("=" * 80)

    # Raw metrics
    print(f"\n{'=' * 80}")
    print(
        f"{'RAW (no MMR)':>30s} {'NDCG@40':>10s} {'MAP':>10s} {'Brier':>10s} {'MedRank':>10s}"
    )
    print("-" * 70)
    for var_name in [v[0] for v in variants]:
        fr = all_results[var_name]
        means = {k: float(np.mean([r["raw"][k] for r in fr])) for k in metric_keys}
        print(
            f"{var_name:30s} {means['ndcg_at_40']:10.4f} {means['map']:10.4f} "
            f"{means['brier_up']:10.4f} {means['median_rank']:8.1f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    _main()
