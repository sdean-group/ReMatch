# ReMatch/rematch/optimal_transport/optimal_transport.py
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from .utils import plot_ot_result         
import json 
import numpy as np
import torch

from .fit_pca import load_pca_model, project_weights_to_pixel_space
from .utils import save_ot_dataset_as_nc
from .optimal_transport_utils import (
    ensure_dir,
    load_training_data,
    load_weight_pair_dataset2,
    flatten_condition_features,
    get_channel_weights,
    build_augmented_features,
    blockwise_topk_sqdist,
)

# ============================================================
# Diagnostics
# ============================================================
def compute_target_usage(
    knn_idx: np.ndarray,      # (N, k)
    plan_vals: np.ndarray,    # (N, k)
    n_target: int,
) -> np.ndarray:
    """Column marginal on sparse graph."""
    usage = np.zeros(n_target, dtype=np.float64)
    for i in range(knn_idx.shape[0]):
        np.add.at(usage, knn_idx[i], plan_vals[i].astype(np.float64))
    return usage


def summarize_marginal_error(
    row_sum: np.ndarray,
    col_sum: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Dict[str, float]:
    row_err = row_sum - a
    col_err = col_sum - b
    return {
        "row_l1": float(np.sum(np.abs(row_err))),
        "row_linf": float(np.max(np.abs(row_err))),
        "row_rel_linf": float(np.max(np.abs(row_err) / (a + 1e-12))),
        "col_l1": float(np.sum(np.abs(col_err))),
        "col_linf": float(np.max(np.abs(col_err))),
        "col_rel_linf": float(np.max(np.abs(col_err) / (b + 1e-12))),
        "total_plan_mass": float(row_sum.sum()),
        "total_source_mass": float(a.sum()),
        "total_target_mass": float(b.sum()),
        "n_uncovered_targets": int(np.sum(col_sum <= 1e-20)),
    }


def summarize_usage(usage: np.ndarray) -> Dict[str, float]:
    usage = usage.astype(np.float64)
    total = max(float(usage.sum()), 1e-12)
    sorted_usage = np.sort(usage)[::-1]
    return {
        "usage_min": float(usage.min()),
        "usage_max": float(usage.max()),
        "usage_mean": float(usage.mean()),
        "usage_std": float(usage.std()),
        "top1_frac": float(sorted_usage[:1].sum() / total),
        "top5_frac": float(sorted_usage[:5].sum() / total),
        "top10_frac": float(sorted_usage[:10].sum() / total),
        "n_unused_targets": int(np.sum(usage <= 1e-20)),
    }


def summarize_selected_count(selected_idx: np.ndarray, n_target: int) -> Dict[str, float]:
    flat = selected_idx.reshape(-1)
    flat = flat[flat >= 0]
    counts = np.bincount(flat, minlength=n_target).astype(np.int64)

    total = max(int(counts.sum()), 1)
    sorted_counts = np.sort(counts)[::-1]

    return {
        "selected_count_min": int(counts.min()),
        "selected_count_max": int(counts.max()),
        "selected_count_mean": float(counts.mean()),
        "selected_count_std": float(counts.std()),
        "selected_top1_frac": float(sorted_counts[:1].sum() / total),
        "selected_top5_frac": float(sorted_counts[:5].sum() / total),
        "selected_top10_frac": float(sorted_counts[:10].sum() / total),
        "n_targets_with_0_selected": int(np.sum(counts == 0)),
        "n_targets_with_1_selected": int(np.sum(counts == 1)),
        "n_targets_with_ge_10_selected": int(np.sum(counts >= 10)),
    }


# ============================================================
# Sparse graph coverage repair
# ============================================================
def _replace_or_insert_edge(
    knn_idx: np.ndarray,
    knn_cost: np.ndarray,
    src_i: int,
    tgt_j: int,
    cost_ij: float,
) -> None:
    """
    Force source row src_i to contain target tgt_j in its sparse support.
    If tgt_j already exists, keep smaller cost. Otherwise replace worst edge.
    """
    row = knn_idx[src_i]
    hit = np.where(row == tgt_j)[0]
    if hit.size > 0:
        h = int(hit[0])
        knn_cost[src_i, h] = min(float(knn_cost[src_i, h]), float(cost_ij))
        return

    pos = int(np.argmax(knn_cost[src_i]))
    knn_idx[src_i, pos] = int(tgt_j)
    knn_cost[src_i, pos] = float(cost_ij)


def ensure_target_min_indegree(
    f_source: np.ndarray,
    f_target: np.ndarray,
    knn_idx: np.ndarray,
    knn_cost: np.ndarray,
    min_indegree: int = 2,
    x_block: int = 512,
    y_block: int = 2048,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Ensure each target has at least min_indegree incoming source rows in sparse graph.
    """
    N, _ = knn_idx.shape
    M = f_target.shape[0]

    if min_indegree < 1:
        return knn_idx, knn_cost, {
            "n_targets_initial_undercovered": 0,
            "n_edges_inserted_or_replaced": 0,
            "min_indegree_required": int(min_indegree),
        }

    indegree = np.bincount(knn_idx.reshape(-1), minlength=M)
    under = np.where(indegree < min_indegree)[0]

    if under.size == 0:
        return knn_idx, knn_cost, {
            "n_targets_initial_undercovered": 0,
            "n_edges_inserted_or_replaced": 0,
            "min_indegree_required": int(min_indegree),
        }

    rev_k = min(max(min_indegree, 1), N)
    rev_idx, rev_cost = blockwise_topk_sqdist(
        X=f_target,
        Y=f_source,
        k=rev_k,
        x_block=y_block,
        y_block=x_block,
        device=device,
    )

    inserted = 0
    for tgt_j in under:
        current = int(indegree[tgt_j])
        need = max(0, min_indegree - current)
        for q in range(min(need, rev_k)):
            src_i = int(rev_idx[tgt_j, q])
            cost_ij = float(rev_cost[tgt_j, q])
            _replace_or_insert_edge(knn_idx, knn_cost, src_i, int(tgt_j), cost_ij)
            inserted += 1

    order = np.argsort(knn_cost, axis=1)
    knn_idx = np.take_along_axis(knn_idx, order, axis=1).astype(np.int32)
    knn_cost = np.take_along_axis(knn_cost, order, axis=1).astype(np.float32)

    return knn_idx, knn_cost, {
        "n_targets_initial_undercovered": int(under.size),
        "n_edges_inserted_or_replaced": int(inserted),
        "min_indegree_required": int(min_indegree),
    }


# ============================================================
# Balanced sparse Sinkhorn
# ============================================================
def sparse_balanced_sinkhorn_knn(
    knn_idx: np.ndarray,
    knn_cost: np.ndarray,
    n_target: int,
    reg: float = 0.1,
    n_iter: int = 1500,
    tol: float = 1e-8,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    eps: float = 1e-300,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Balanced entropic Sinkhorn on fixed sparse source->target support.
    """
    N, _ = knn_idx.shape

    if a is None:
        a = np.ones(N, dtype=np.float64) / float(N)
    else:
        a = np.asarray(a, dtype=np.float64)
        a = a / a.sum()

    if b is None:
        b = np.ones(n_target, dtype=np.float64) / float(n_target)
    else:
        b = np.asarray(b, dtype=np.float64)
        b = b / b.sum()

    if not np.isclose(a.sum(), b.sum(), rtol=1e-10, atol=1e-12):
        raise ValueError(f"Balanced OT requires equal total mass, got {a.sum()} and {b.sum()}.")

    K = np.exp(-knn_cost.astype(np.float64) / float(reg))
    K = np.maximum(K, eps)

    u = np.ones(N, dtype=np.float64)
    v = np.ones(n_target, dtype=np.float64)
    prev_u = u.copy()

    rel = np.inf
    for it in range(int(n_iter)):
        Kv = (K * v[knn_idx]).sum(axis=1) + eps
        u = a / Kv

        KTu = np.zeros(n_target, dtype=np.float64)
        np.add.at(KTu, knn_idx.reshape(-1), (K * u[:, None]).reshape(-1))
        KTu = KTu + eps
        v = b / KTu

        rel = np.max(np.abs(u - prev_u) / (np.abs(prev_u) + 1e-12))
        prev_u = u.copy()
        if rel < tol:
            break

    plan_vals = (u[:, None] * K) * v[knn_idx]
    row_sum = plan_vals.sum(axis=1)
    col_sum = compute_target_usage(knn_idx, plan_vals, n_target=n_target)

    diag = summarize_marginal_error(row_sum=row_sum, col_sum=col_sum, a=a, b=b)
    diag.update({
        "sinkhorn_iter_used": int(it + 1),
        "sinkhorn_rel_u": float(rel),
        "reg": float(reg),
    })
    return plan_vals.astype(np.float32), diag


# ============================================================
# Pass-1 selected-count-aware candidate penalty
# ============================================================
def build_candidate_penalty_from_selected_count(
    selected_idx: np.ndarray,
    n_target: int,
    penalty_eta: float = 1.0,
    penalty_beta: float = 1.0,
    hard_cap: Optional[int] = None,
    hard_cap_penalty: float = 5.0,
) -> np.ndarray:
    """
    Convert provisional selected-count into per-target candidate penalty.

    penalty_j = penalty_eta * (count_j / mean_count)^penalty_beta
    If hard_cap is set and count_j >= hard_cap, add hard_cap_penalty.
    """
    flat = selected_idx.reshape(-1)
    flat = flat[flat >= 0]
    counts = np.bincount(flat, minlength=n_target).astype(np.float64)

    mean_count = max(float(counts.mean()), 1e-12)
    count_norm = counts / mean_count
    penalty = penalty_eta * np.power(count_norm, penalty_beta)

    if hard_cap is not None:
        penalty = penalty + hard_cap_penalty * (counts >= hard_cap).astype(np.float64)

    return penalty.astype(np.float32)


def apply_target_penalty_to_knn_cost(
    knn_idx: np.ndarray,
    knn_cost_scaled: np.ndarray,
    per_target_penalty: np.ndarray,
) -> np.ndarray:
    """
    Add target-wise penalty to sparse candidate costs.
    """
    return knn_cost_scaled + per_target_penalty[knn_idx].astype(np.float32)


# ============================================================
# Final top-m selection with hard cap
# ============================================================
def weighted_topm_from_sparse_plan_hardcap(
    plan_idx: np.ndarray,
    plan_vals: np.ndarray,
    target_weights: np.ndarray,
    knn_cost: Optional[np.ndarray] = None,
    n_target: Optional[int] = None,
    top_m: int = 3,
    max_selected_per_target: Optional[int] = None,
    selected_cost_tradeoff_gamma: float = 0.0,
    strict_cap: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Final reconstruction stage.

    - chooses up to top_m targets per source
    - respects hard cap as much as possible
    - if strict_cap=True, targets over cap are not used in normal selection
    - if all candidates blocked, fallback uses least-violating option
    """
    N, k = plan_idx.shape
    Kc = target_weights.shape[1]

    if n_target is None:
        n_target = int(plan_idx.max()) + 1
    if top_m < 1:
        raise ValueError("top_m must be >= 1.")
    if top_m > k:
        raise ValueError(f"top_m={top_m} must be <= knn_k={k}.")

    transported = np.zeros((N, Kc), dtype=np.float32)
    selected_idx = np.full((N, top_m), -1, dtype=np.int32)
    selected_vals = np.zeros((N, top_m), dtype=np.float32)
    selected_count = np.zeros(n_target, dtype=np.int32)

    sorted_vals = np.sort(plan_vals, axis=1)
    confidence = sorted_vals[:, -1] - sorted_vals[:, -2] if k >= 2 else sorted_vals[:, -1]
    source_order = np.argsort(-confidence)

    n_fallback_sources = 0
    n_cap_violations = 0
    n_underfilled_sources = 0

    for i in source_order:
        available_pos = list(range(k))
        chosen_pos = []

        for _ in range(top_m):
            if len(available_pos) == 0:
                break

            candidate_positions = []
            candidate_scores = []

            for pos in available_pos:
                tgt = int(plan_idx[i, pos])

                if strict_cap and max_selected_per_target is not None:
                    if selected_count[tgt] >= max_selected_per_target:
                        continue

                score = float(plan_vals[i, pos])
                if knn_cost is not None and selected_cost_tradeoff_gamma > 0.0:
                    score = score / ((1.0 + float(knn_cost[i, pos])) ** float(selected_cost_tradeoff_gamma))

                candidate_positions.append(pos)
                candidate_scores.append(score)

            if len(candidate_positions) == 0:
                # fallback only when all candidates blocked
                n_fallback_sources += 1
                for pos in available_pos:
                    score = float(plan_vals[i, pos])
                    if knn_cost is not None and selected_cost_tradeoff_gamma > 0.0:
                        score = score / ((1.0 + float(knn_cost[i, pos])) ** float(selected_cost_tradeoff_gamma))
                    candidate_positions.append(pos)
                    candidate_scores.append(score)

                if len(candidate_positions) == 0:
                    break

            best_local = int(np.argmax(np.asarray(candidate_scores, dtype=np.float64)))
            best_pos = int(candidate_positions[best_local])
            best_tgt = int(plan_idx[i, best_pos])

            if max_selected_per_target is not None and selected_count[best_tgt] >= max_selected_per_target:
                n_cap_violations += 1

            chosen_pos.append(best_pos)
            selected_count[best_tgt] += 1
            available_pos.remove(best_pos)

        if len(chosen_pos) < top_m:
            n_underfilled_sources += 1

        if len(chosen_pos) == 0:
            continue

        chosen_targets = plan_idx[i, chosen_pos].astype(np.int32)
        vals = plan_vals[i, chosen_pos].astype(np.float64)
        vals_sum = vals.sum()

        if vals_sum <= eps:
            vals = np.ones(len(chosen_pos), dtype=np.float64) / len(chosen_pos)
        else:
            vals = vals / vals_sum

        selected_idx[i, :len(chosen_pos)] = chosen_targets
        selected_vals[i, :len(chosen_pos)] = vals.astype(np.float32)
        transported[i] = (vals[:, None].astype(np.float32) * target_weights[chosen_targets]).sum(axis=0)

    summary = summarize_selected_count(selected_idx, n_target=n_target)
    summary.update({
        "top_m": int(top_m),
        "max_selected_per_target": None if max_selected_per_target is None else int(max_selected_per_target),
        "selected_cost_tradeoff_gamma": float(selected_cost_tradeoff_gamma),
        "strict_cap": bool(strict_cap),
        "n_fallback_sources": int(n_fallback_sources),
        "n_cap_violations": int(n_cap_violations),
        "n_underfilled_sources": int(n_underfilled_sources),
    })
    return transported, selected_idx, selected_vals, summary


# ============================================================
# Provisional selection for pass-1 count estimate
# ============================================================
def provisional_topm_selection(
    plan_idx: np.ndarray,
    plan_vals: np.ndarray,
    n_target: int,
    top_m: int = 3,
) -> np.ndarray:
    """
    Fast provisional top-m selection using largest plan values only.
    Used only to estimate which targets are being repeatedly selected.
    """
    N, k = plan_idx.shape
    selected_idx = np.full((N, top_m), -1, dtype=np.int32)

    for i in range(N):
        vals = plan_vals[i]
        keep = np.argsort(vals)[-min(top_m, k):][::-1]
        chosen = plan_idx[i, keep].astype(np.int32)
        selected_idx[i, :len(chosen)] = chosen

    return selected_idx


# ============================================================
# Channel-wise OT with 2-pass candidate penalty
# ============================================================
def solve_one_channel_sparse_balanced_map(
    residual_source: np.ndarray,
    residual_target: np.ndarray,
    cond_source: np.ndarray,
    cond_target: np.ndarray,
    n_modes_residual: np.ndarray,
    n_modes_cond: np.ndarray,
    channel: int,
    alpha: float,
    lambda_cond: float,
    knn_k: int,
    top_m: int,
    pass1_penalty_eta: float,
    pass1_penalty_beta: float,
    candidate_hard_cap: Optional[int],
    candidate_hard_cap_penalty: float,
    max_selected_per_target: Optional[int],
    selected_cost_tradeoff_gamma: float,
    enforce_target_min_indegree: bool,
    reg: float,
    sinkhorn_iter: int,
    sinkhorn_tol: float,
    device: str,
    x_block: int,
    y_block: int,
    use_cond_modes_per_channel: Optional[Dict[int, int]] = None,
):
    w1_raw, w2_raw, w1_norm, w2_norm, residual_stats = get_channel_weights(
        residual_source, residual_target, channel, n_modes_residual
    )

    z1, z2, condition_stats = flatten_condition_features(
        cond_source,
        cond_target,
        n_modes_cond,
        use_modes_per_channel=use_cond_modes_per_channel,
        stats_from_target=True,
    )

    f1, f2 = build_augmented_features(
        w1c=w1_norm,
        w2c=w2_norm,
        z1=z1,
        z2=z2,
        alpha=alpha,
        lambda_cond=lambda_cond,
    )

    N = f1.shape[0]
    M = f2.shape[0]
    source_to_target_ratio = float(N) / float(M)

    knn_idx, knn_cost = blockwise_topk_sqdist(
        X=f1,
        Y=f2,
        k=knn_k,
        x_block=x_block,
        y_block=y_block,
        device=device,
    )

    coverage_info = {
        "n_targets_initial_undercovered": 0,
        "n_edges_inserted_or_replaced": 0,
        "min_indegree_required": 0,
    }
    if enforce_target_min_indegree:
        min_indegree = max(1, int(np.ceil(source_to_target_ratio)))
        knn_idx, knn_cost, coverage_info = ensure_target_min_indegree(
            f_source=f1,
            f_target=f2,
            knn_idx=knn_idx,
            knn_cost=knn_cost,
            min_indegree=min_indegree,
            x_block=x_block,
            y_block=y_block,
            device=device,
        )

    positive = knn_cost[knn_cost > 0]
    if positive.size == 0:
        raise ValueError("All kNN costs are zero.")
    cost_scale = np.median(positive)
    knn_cost_scaled = knn_cost / max(float(cost_scale), 1e-12)

    a = np.ones(N, dtype=np.float64) / float(N)
    b = np.ones(M, dtype=np.float64) / float(M)

    # --------------------------------------------------------
    # Pass 1: baseline balanced Sinkhorn
    # --------------------------------------------------------
    plan_vals_pass1, sinkhorn_diag_pass1 = sparse_balanced_sinkhorn_knn(
        knn_idx=knn_idx,
        knn_cost=knn_cost_scaled,
        n_target=M,
        reg=reg,
        n_iter=sinkhorn_iter,
        tol=sinkhorn_tol,
        a=a,
        b=b,
    )

    usage_pass1 = compute_target_usage(knn_idx, plan_vals_pass1, n_target=M)
    usage_summary_pass1 = summarize_usage(usage_pass1)

    provisional_selected_idx = provisional_topm_selection(
        plan_idx=knn_idx,
        plan_vals=plan_vals_pass1,
        n_target=M,
        top_m=top_m,
    )
    provisional_selected_summary = summarize_selected_count(provisional_selected_idx, n_target=M)

    # --------------------------------------------------------
    # Candidate-stage penalty from pass-1 selected count
    # --------------------------------------------------------
    per_target_penalty = build_candidate_penalty_from_selected_count(
        selected_idx=provisional_selected_idx,
        n_target=M,
        penalty_eta=pass1_penalty_eta,
        penalty_beta=pass1_penalty_beta,
        hard_cap=candidate_hard_cap,
        hard_cap_penalty=candidate_hard_cap_penalty,
    )

    knn_cost_refined = apply_target_penalty_to_knn_cost(
        knn_idx=knn_idx,
        knn_cost_scaled=knn_cost_scaled,
        per_target_penalty=per_target_penalty,
    )

    # --------------------------------------------------------
    # Pass 2: refined balanced Sinkhorn
    # --------------------------------------------------------
    plan_vals_pass2, sinkhorn_diag_pass2 = sparse_balanced_sinkhorn_knn(
        knn_idx=knn_idx,
        knn_cost=knn_cost_refined,
        n_target=M,
        reg=reg,
        n_iter=sinkhorn_iter,
        tol=sinkhorn_tol,
        a=a,
        b=b,
    )

    usage_pass2 = compute_target_usage(knn_idx, plan_vals_pass2, n_target=M)
    usage_summary_pass2 = summarize_usage(usage_pass2)

    # --------------------------------------------------------
    # Final hard-cap top-m reconstruction
    # --------------------------------------------------------
    transported_weights, selected_idx, selected_vals, selected_summary = (
        weighted_topm_from_sparse_plan_hardcap(
            plan_idx=knn_idx,
            plan_vals=plan_vals_pass2,
            target_weights=w2_raw,
            knn_cost=knn_cost_refined,
            n_target=M,
            top_m=top_m,
            max_selected_per_target=max_selected_per_target,
            selected_cost_tradeoff_gamma=selected_cost_tradeoff_gamma,
            strict_cap=True,
        )
    )

    print(
        f"[channel {channel}] "
        f"usage_top10 pass1={usage_summary_pass1['top10_frac']:.4f}, "
        f"pass2={usage_summary_pass2['top10_frac']:.4f}, "
        f"selected_top10={selected_summary['selected_top10_frac']:.4f}, "
        f"selected_max={selected_summary['selected_count_max']}, "
        f"transported_std={transported_weights.std():.6f}"
    )

    return {
        "channel": channel,
        "transported_weights": transported_weights,
        "selected_idx": selected_idx,
        "selected_vals": selected_vals,
        "provisional_selected_idx": provisional_selected_idx,
        "knn_idx": knn_idx,
        "knn_cost_scaled": knn_cost_scaled.astype(np.float32),
        "knn_cost_refined": knn_cost_refined.astype(np.float32),
        "plan_vals_pass1": plan_vals_pass1,
        "plan_vals": plan_vals_pass2,
        "usage_pass1": usage_pass1,
        "usage": usage_pass2,
        "per_target_penalty": per_target_penalty,
        "residual_stats": residual_stats,
        "condition_stats": condition_stats,
        "usage_summary_pass1": usage_summary_pass1,
        "usage_summary": usage_summary_pass2,
        "provisional_selected_summary": provisional_selected_summary,
        "selected_summary": selected_summary,
        "sinkhorn_diag_pass1": sinkhorn_diag_pass1,
        "sinkhorn_diag": sinkhorn_diag_pass2,
        "coverage_info": coverage_info,
        "source_to_target_ratio": source_to_target_ratio,
        "cost_scale": float(cost_scale),
    }


def _channel_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["local_gpu"])
    torch.cuda.set_device(0)

    return solve_one_channel_sparse_balanced_map(
        residual_source=args["residual_source"],
        residual_target=args["residual_target"],
        cond_source=args["cond_source"],
        cond_target=args["cond_target"],
        n_modes_residual=args["n_modes_residual"],
        n_modes_cond=args["n_modes_cond"],
        channel=args["channel"],
        alpha=args["alpha"],
        lambda_cond=args["lambda_cond"],
        knn_k=args["knn_k"],
        top_m=args["top_m"],
        pass1_penalty_eta=args["pass1_penalty_eta"],
        pass1_penalty_beta=args["pass1_penalty_beta"],
        candidate_hard_cap=args["candidate_hard_cap"],
        candidate_hard_cap_penalty=args["candidate_hard_cap_penalty"],
        max_selected_per_target=args["max_selected_per_target"],
        selected_cost_tradeoff_gamma=args["selected_cost_tradeoff_gamma"],
        reg=args["reg"],
        sinkhorn_iter=args["sinkhorn_iter"],
        sinkhorn_tol=args["sinkhorn_tol"],
        device="cuda",
        x_block=args["x_block"],
        y_block=args["y_block"],
        use_cond_modes_per_channel=args["use_cond_modes_per_channel"],
        enforce_target_min_indegree=args["enforce_target_min_indegree"],
    )


# ============================================================
# Full multi-GPU pipeline
# ============================================================
def run_channelwise_sparse_balanced_map_multigpu(
    residual_npz_path: str,
    condition_npz_path: str,
    pca_model,
    x_lr: np.ndarray,
    x_lr_full: np.ndarray,
    x_gt: np.ndarray,
    save_dir: str,
    final_nc_path: str,
    channel_names: List[str],
    alpha: float = 1.0,
    lambda_cond: float = 1.0,
    knn_k: int = 32,
    top_m: int = 3,
    pass1_penalty_eta: float = 1.0,
    pass1_penalty_beta: float = 1.0,
    candidate_hard_cap: Optional[int] = None,
    candidate_hard_cap_penalty: float = 5.0,
    max_selected_per_target: Optional[int] = None,
    selected_cost_tradeoff_gamma: float = 0.15,
    reg: float = 0.1,
    sinkhorn_iter: int = 1500,
    sinkhorn_tol: float = 1e-8,
    gpu_ids: Tuple[int, ...] = (0, 1, 2, 3),
    x_block: int = 512,
    y_block: int = 2048,
    save_sparse_plan: bool = True,
    use_cond_modes_per_channel: Optional[Dict[int, int]] = None,
    enforce_target_min_indegree: bool = True,
):
    save_dir = ensure_dir(save_dir)
    print(f"loading weight pair dataset from {residual_npz_path} and {condition_npz_path}")

    res_source, res_target, _, res_n_modes = load_weight_pair_dataset2(residual_npz_path)
    cond_source, cond_target, _, cond_n_modes = load_weight_pair_dataset2(condition_npz_path)

    print(f"res_source shape: {res_source.shape}")
    print(f"res_target shape: {res_target.shape}")
    print(f"cond_source shape: {cond_source.shape}")
    print(f"cond_target shape: {cond_target.shape}")
    print(f"res_n_modes: {res_n_modes}")
    print(f"cond_n_modes: {cond_n_modes}")

    T1, C, _ = res_source.shape
    T2 = res_target.shape[0]
    source_to_target_ratio = float(T1) / float(T2)

    if max_selected_per_target is None:
        avg_selected = int(np.ceil((T1 * top_m) / T2))
        max_selected_per_target = max(2, avg_selected + 2)

    if candidate_hard_cap is None:
        candidate_hard_cap = max_selected_per_target + 2

    transported_all = np.zeros_like(res_source, dtype=np.float32)

    job_args = []
    for c in range(C):
        local_gpu = gpu_ids[c % len(gpu_ids)]
        job_args.append({
            "local_gpu": local_gpu,
            "residual_source": res_source,
            "residual_target": res_target,
            "cond_source": cond_source,
            "cond_target": cond_target,
            "n_modes_residual": res_n_modes,
            "n_modes_cond": cond_n_modes,
            "channel": c,
            "alpha": alpha,
            "lambda_cond": lambda_cond,
            "knn_k": knn_k,
            "top_m": top_m,
            "pass1_penalty_eta": pass1_penalty_eta,
            "pass1_penalty_beta": pass1_penalty_beta,
            "candidate_hard_cap": candidate_hard_cap,
            "candidate_hard_cap_penalty": candidate_hard_cap_penalty,
            "max_selected_per_target": max_selected_per_target,
            "selected_cost_tradeoff_gamma": selected_cost_tradeoff_gamma,
            "reg": reg,
            "sinkhorn_iter": sinkhorn_iter,
            "sinkhorn_tol": sinkhorn_tol,
            "x_block": x_block,
            "y_block": y_block,
            "use_cond_modes_per_channel": use_cond_modes_per_channel,
            "enforce_target_min_indegree": enforce_target_min_indegree,
        })

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(gpu_ids)) as pool:
        print(f"running {len(job_args)} channels on {len(gpu_ids)} GPUs")
        results = pool.map(_channel_worker, job_args)

    usage_summary_pass1_all = {}
    usage_summary_all = {}
    provisional_selected_summary_all = {}
    selected_summary_all = {}
    sinkhorn_diag_pass1_all = {}
    sinkhorn_diag_all = {}
    coverage_info_all = {}

    for out in results:
        c = out["channel"]
        kc = out["residual_stats"]["Kc"]
        transported_all[:, c, :kc] = out["transported_weights"]

        usage_summary_pass1_all[f"channel_{c}"] = out["usage_summary_pass1"]
        usage_summary_all[f"channel_{c}"] = out["usage_summary"]
        provisional_selected_summary_all[f"channel_{c}"] = out["provisional_selected_summary"]
        selected_summary_all[f"channel_{c}"] = out["selected_summary"]
        sinkhorn_diag_pass1_all[f"channel_{c}"] = out["sinkhorn_diag_pass1"]
        sinkhorn_diag_all[f"channel_{c}"] = out["sinkhorn_diag"]
        coverage_info_all[f"channel_{c}"] = out["coverage_info"]

        if save_sparse_plan:
            np.savez_compressed(
                Path(save_dir) / f"channel_{c}_balanced_sparse_plan.npz",
                knn_idx=out["knn_idx"],
                knn_cost_scaled=out["knn_cost_scaled"],
                knn_cost_refined=out["knn_cost_refined"],
                plan_vals_pass1=out["plan_vals_pass1"],
                plan_vals=out["plan_vals"],
                provisional_selected_idx=out["provisional_selected_idx"],
                selected_idx=out["selected_idx"],
                selected_vals=out["selected_vals"],
                usage_pass1=out["usage_pass1"],
                usage=out["usage"],
                per_target_penalty=out["per_target_penalty"],
                channel=np.array(c, dtype=np.int32),
                Kc=np.array(kc, dtype=np.int32),
                cost_scale=np.array(out["cost_scale"], dtype=np.float32),
            )

    np.save(Path(save_dir) / "transported_source_residual_weights.npy", transported_all)

    x_res_ot = project_weights_to_pixel_space(
        weights=transported_all,
        pca_model=pca_model,
        n_modes_per_channel=res_n_modes,
    )
    np.save(Path(save_dir) / "source_residual_ot_reconstructed.npy", x_res_ot)

    meta = {
        "method": "balanced_ot_two_pass_candidate_penalty_plus_hardcap_topm",
        "alpha": alpha,
        "lambda_cond": lambda_cond,
        "knn_k": knn_k,
        "top_m": top_m,
        "pass1_penalty_eta": pass1_penalty_eta,
        "pass1_penalty_beta": pass1_penalty_beta,
        "candidate_hard_cap": candidate_hard_cap,
        "candidate_hard_cap_penalty": candidate_hard_cap_penalty,
        "max_selected_per_target": max_selected_per_target,
        "selected_cost_tradeoff_gamma": selected_cost_tradeoff_gamma,
        "reg": reg,
        "sinkhorn_iter": sinkhorn_iter,
        "sinkhorn_tol": sinkhorn_tol,
        "gpu_ids": list(gpu_ids),
        "x_block": x_block,
        "y_block": y_block,
        "enforce_target_min_indegree": enforce_target_min_indegree,
        "source_to_target_ratio": source_to_target_ratio,
        "residual_npz_path": residual_npz_path,
        "condition_npz_path": condition_npz_path,
        "final_nc_path": final_nc_path,
        "usage_summary_pass1": usage_summary_pass1_all,
        "usage_summary": usage_summary_all,
        "provisional_selected_summary": provisional_selected_summary_all,
        "selected_summary": selected_summary_all,
        "sinkhorn_diag_pass1": sinkhorn_diag_pass1_all,
        "sinkhorn_diag": sinkhorn_diag_all,
        "coverage_info": coverage_info_all,
    }
    with open(Path(save_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return transported_all, x_res_ot

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_npz_path", type=str, default="")
    parser.add_argument("--condition_npz_path", type=str, default="")
    parser.add_argument("--path_pca_model", type=str, default="")
    parser.add_argument("--source_nc_path", type=str, default="")
    parser.add_argument("--target_nc_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--channel_names", type=list, default=["50u", "50v", "75u", "75v", "100u", "100v", "125u", "125v", "150u", "150v"])
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--gpu_ids", type=str, default=0)
    return parser.parse_args()
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    residual_npz_path = args.residual_npz_path
    condition_npz_path = args.condition_npz_path
    path_pca_model = args.path_pca_model
    source_nc_path = args.source_nc_path
    target_nc_path = args.target_nc_path
    save_dir = args.save_dir
    channel_names = args.channel_names
    gpu_ids=args.gpu_ids
   
    # knn_k = 12
    knn_k = 36
    top_m = 3

    alpha = 1.0
    lambda_cond = 1.5
    reg = 0.1
    sinkhorn_iter = 1500

    # pass-1 selected-count -> candidate penalty
    pass1_penalty_eta = 0.75
    pass1_penalty_beta = 1.25
    candidate_hard_cap = 18
    candidate_hard_cap_penalty = 4.0

    # final hard-cap selection
    max_selected_per_target = 3
    selected_cost_tradeoff_gamma = 0.15

    
    final_nc_path = os.path.join(save_dir, "reg_2018-2019_ot.nc")
    final_all_nc_path = os.path.join(save_dir, "reg_2018-2020_ot.nc")

    x_lr_full, x_lr, x_res_source, x_gt = load_training_data(source_nc_path)
    _, _, x_res_target, _ = load_training_data(target_nc_path)
    pca_model = load_pca_model(path_pca_model)




    transported_all, x_res_ot = run_channelwise_sparse_balanced_map_multigpu(
        residual_npz_path=residual_npz_path,
        condition_npz_path=condition_npz_path,
        pca_model=pca_model,
        x_lr=x_lr,
        x_lr_full=x_lr_full,
        x_gt=x_gt,
        save_dir=save_dir,
        final_nc_path=final_nc_path,
        channel_names=channel_names,
        alpha=alpha,
        lambda_cond=lambda_cond,
        knn_k=knn_k,
        top_m=top_m,
        pass1_penalty_eta=pass1_penalty_eta,
        pass1_penalty_beta=pass1_penalty_beta,
        candidate_hard_cap=candidate_hard_cap,
        candidate_hard_cap_penalty=candidate_hard_cap_penalty,
        max_selected_per_target=max_selected_per_target,
        selected_cost_tradeoff_gamma=selected_cost_tradeoff_gamma,
        reg=reg,
        sinkhorn_iter=sinkhorn_iter,
        sinkhorn_tol=1e-8,
        gpu_ids=gpu_ids,
        x_block=512,
        y_block=2048,
        save_sparse_plan=True,
        enforce_target_min_indegree=True,
    )

    save_ot_dataset_as_nc(
        source_nc_path=source_nc_path,
        target_nc_path=target_nc_path,
        save_ot_reg_path=final_nc_path,
        save_ot_all_reg_path=final_all_nc_path,
        x_res_ot=x_res_ot,
        channel_names=channel_names,
        compress_level=4,
    )

    x_res_ot = x_res_ot.squeeze(axis=0)

    res_mean = np.mean(x_res_source, axis=(1, 2, 3))
    res_std = np.std(x_res_source, axis=(1, 2, 3))
    res_ot_mean = np.mean(x_res_ot, axis=(1, 2, 3))
    res_ot_std = np.std(x_res_ot, axis=(1, 2, 3))

    summary = {
        "res_mean": np.round(res_mean, 6).tolist(),
        "res_ot_mean": np.round(res_ot_mean, 6).tolist(),
        "res_std": np.round(res_std, 6).tolist(),
        "res_ot_std": np.round(res_ot_std, 6).tolist(),
    }

    # plot_dir = (
    #     f"./plots/{model}/"
    #     f"knn{knn_k}_topm{top_m}_alpha{alpha}_lambda{lambda_cond}"
    #     f"_reg{reg}_peta{pass1_penalty_eta}_pbeta{pass1_penalty_beta}"
    #     f"_candcap{candidate_hard_cap}_finalcap{max_selected_per_target}"
    # )
    # os.makedirs(plot_dir, exist_ok=True)

    # with open(f"{plot_dir}/residual_summary.json", "w") as f:
    #     json.dump(summary, f, indent=4)

    # plot_ot_result(
    #     res_ot=x_res_ot,
    #     res_original=x_res_source,
    #     save_file=plot_dir,
    #     channel_names=channel_names,
    #     # channel_idx=[0,1,2,3],
    #     # idx=[0,50,150,200],
    # )
