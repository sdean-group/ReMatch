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
# Balanced sparse Sinkhorn helpers
# ============================================================
def compute_target_usage(
    knn_idx: np.ndarray,
    plan_vals: np.ndarray,
    n_target: int,
) -> np.ndarray:
    """Column marginal of a sparse plan."""
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
    }


def _replace_or_insert_edge(
    knn_idx: np.ndarray,
    knn_cost: np.ndarray,
    src_i: int,
    tgt_j: int,
    cost_ij: float,
) -> None:
    """
    Ensure edge src_i -> tgt_j exists in a fixed-width sparse KNN graph.

    If the edge is already present, keep the smaller cost. Otherwise, replace the
    largest-cost edge in that source row. This keeps KNN width fixed.
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
    Balanced sparse OT can be infeasible if some target columns have too few
    incoming source edges. With N_source = 2 * N_target and uniform masses,
    each target needs mass equal to two source rows. Therefore each target
    should have at least two incoming source candidates.

    This function augments the source->target KNN graph by adding each
    under-covered target to its nearest source rows. The sparse width is kept
    fixed by replacing the largest-cost edge in the selected source rows.
    """
    N, k = knn_idx.shape
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

    # Reverse nearest sources for all targets. We need at least min_indegree
    # candidate source rows per target.
    rev_k = min(max(min_indegree, 1), N)
    rev_idx, rev_cost = blockwise_topk_sqdist(
        X=f_target,
        Y=f_source,
        k=rev_k,
        x_block=y_block,
        y_block=x_block,
        device=device,
    )
    # rev_idx[j, q] is a source index nearest to target j.

    inserted = 0
    for tgt_j in under:
        current = int(indegree[tgt_j])
        need = max(0, min_indegree - current)
        if need == 0:
            continue
        for q in range(min(need, rev_k)):
            src_i = int(rev_idx[tgt_j, q])
            cost_ij = float(rev_cost[tgt_j, q])
            _replace_or_insert_edge(knn_idx, knn_cost, src_i, int(tgt_j), cost_ij)
            inserted += 1

    # Sort each row by cost for cleaner saved plans.
    order = np.argsort(knn_cost, axis=1)
    knn_idx = np.take_along_axis(knn_idx, order, axis=1).astype(np.int32)
    knn_cost = np.take_along_axis(knn_cost, order, axis=1).astype(np.float32)

    return knn_idx, knn_cost, {
        "n_targets_initial_undercovered": int(under.size),
        "n_edges_inserted_or_replaced": int(inserted),
        "min_indegree_required": int(min_indegree),
    }


def sparse_balanced_sinkhorn_knn(
    knn_idx: np.ndarray,
    knn_cost: np.ndarray,
    n_target: int,
    reg: float = 0.05,
    n_iter: int = 1000,
    tol: float = 1e-8,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    eps: float = 1e-300,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Balanced entropic Sinkhorn on a fixed sparse source->target KNN graph.

    Solves approximately:
        min_pi <pi, C> + reg * sum pi (log pi - 1)
        s.t. pi 1 = a, pi^T 1 = b, pi >= 0

    Note:
        Sparse support may make the exact balanced constraints infeasible.
        Use ensure_target_min_indegree before this function. Diagnostics are
        returned so infeasibility/non-convergence is visible.
    """
    N, k = knn_idx.shape

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


def map_from_sparse_plan_capacity_constrained(
    plan_idx: np.ndarray,
    plan_vals: np.ndarray,
    knn_cost: Optional[np.ndarray] = None,
    n_target: Optional[int] = None,
    max_sources_per_target: int = 3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Deterministic sharp assignment from a sparse transport plan with target capacity.

    Each source is assigned to exactly one target among its sparse candidates.
    Unlike row-wise top-1, this prevents many source samples from collapsing onto
    the same target residual.

    Greedy strategy:
      1. Sort source rows by assignment confidence, high to low.
      2. For each source, try candidate targets in descending plan mass order.
      3. Assign to the first target whose count is below max_sources_per_target.
      4. If all candidates are full, assign to the least-loaded candidate as fallback.

    This keeps residuals sharp because it does not average target residuals.
    """
    N, k = plan_idx.shape
    if n_target is None:
        n_target = int(plan_idx.max()) + 1
    if max_sources_per_target < 1:
        raise ValueError("max_sources_per_target must be >= 1.")

    mapped_idx = np.full(N, -1, dtype=np.int32)
    target_count = np.zeros(n_target, dtype=np.int32)

    # Confidence: how strongly the best candidate dominates the second best.
    sorted_vals = np.sort(plan_vals, axis=1)
    if k >= 2:
        confidence = sorted_vals[:, -1] - sorted_vals[:, -2]
    else:
        confidence = sorted_vals[:, -1]

    # Assign confident rows first, because their best match is more unambiguous.
    source_order = np.argsort(-confidence)

    n_fallback = 0
    n_overflow = 0

    for i in source_order:
        # Prefer larger plan mass. Use lower cost as secondary preference if provided.
        if knn_cost is None:
            cand_order = np.argsort(-plan_vals[i])
        else:
            # Lexsort uses last key as primary. Primary: -plan_vals, secondary: cost.
            cand_order = np.lexsort((knn_cost[i], -plan_vals[i]))

        assigned = False
        for pos in cand_order:
            tgt = int(plan_idx[i, pos])
            if target_count[tgt] < max_sources_per_target:
                mapped_idx[i] = tgt
                target_count[tgt] += 1
                assigned = True
                break

        if not assigned:
            # Sparse graph/capacity conflict fallback. Avoid dropping a source.
            # Choose the least-loaded candidate, then lower cost/higher mass.
            candidates = plan_idx[i, cand_order].astype(np.int32)
            counts = target_count[candidates]
            min_count = counts.min()
            feasible_pos = cand_order[counts == min_count]
            if knn_cost is not None:
                best_local = feasible_pos[np.argmin(knn_cost[i, feasible_pos])]
            else:
                best_local = feasible_pos[np.argmax(plan_vals[i, feasible_pos])]
            tgt = int(plan_idx[i, best_local])
            mapped_idx[i] = tgt
            target_count[tgt] += 1
            n_fallback += 1
            if target_count[tgt] > max_sources_per_target:
                n_overflow += 1

    if np.any(mapped_idx < 0):
        raise RuntimeError("Some source samples were not assigned.")

    summary = {
        "max_sources_per_target": int(max_sources_per_target),
        "map_count_min": int(target_count.min()),
        "map_count_max": int(target_count.max()),
        "map_count_mean": float(target_count.mean()),
        "map_count_std": float(target_count.std()),
        "n_targets_with_0_sources": int(np.sum(target_count == 0)),
        "n_targets_with_1_source": int(np.sum(target_count == 1)),
        "n_targets_with_2_sources": int(np.sum(target_count == 2)),
        "n_targets_with_3_sources": int(np.sum(target_count == 3)),
        "n_targets_with_gt_capacity": int(np.sum(target_count > max_sources_per_target)),
        "n_targets_over_capacity": int(np.sum(target_count > max_sources_per_target)),
        "n_fallback_assignments": int(n_fallback),
        "n_overflow_assignments": int(n_overflow),
    }
    return mapped_idx.astype(np.int32), summary

def weighted_topm_from_sparse_plan_capacity_constrained(
    plan_idx: np.ndarray,
    plan_vals: np.ndarray,
    target_weights: np.ndarray,
    knn_cost: Optional[np.ndarray] = None,
    n_target: Optional[int] = None,
    top_m: int = 3,
    max_sources_per_target: int = 3,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Construct transported residual coefficients by weighted averaging top_m targets
    selected under a target-capacity constraint.

    Each source uses up to top_m target residuals. The candidate pool can be much
    larger, e.g. knn_k=128. Target usage is counted once per selected source-target
    pair, not by transport mass.

    This avoids using all KNN candidates in barycentric reconstruction while also
    preventing a small number of target residuals from being reused too often.
    """
    N, k = plan_idx.shape
    Kc = target_weights.shape[1]

    if n_target is None:
        n_target = int(plan_idx.max()) + 1
    if top_m < 1:
        raise ValueError("top_m must be >= 1.")
    if top_m > k:
        raise ValueError(f"top_m={top_m} must be <= knn_k={k}.")
    if max_sources_per_target < 1:
        raise ValueError("max_sources_per_target must be >= 1.")

    transported = np.zeros((N, Kc), dtype=np.float32)
    selected_idx = np.full((N, top_m), -1, dtype=np.int32)
    selected_vals = np.zeros((N, top_m), dtype=np.float32)

    target_count = np.zeros(n_target, dtype=np.int32)

    # Confidence: how much the strongest candidate dominates.
    sorted_vals = np.sort(plan_vals, axis=1)
    if k >= 2:
        confidence = sorted_vals[:, -1] - sorted_vals[:, -2]
    else:
        confidence = sorted_vals[:, -1]

    # Assign more confident rows first.
    source_order = np.argsort(-confidence)

    n_fallback_sources = 0
    n_underfilled_sources = 0
    n_overflow_assignments = 0

    for i in source_order:
        if knn_cost is None:
            cand_order = np.argsort(-plan_vals[i])
        else:
            # Primary: larger plan value. Secondary: lower cost.
            cand_order = np.lexsort((knn_cost[i], -plan_vals[i]))

        chosen_positions = []

        # First pass: choose candidates below capacity.
        for pos in cand_order:
            tgt = int(plan_idx[i, pos])
            if target_count[tgt] < max_sources_per_target:
                chosen_positions.append(int(pos))
                target_count[tgt] += 1
                if len(chosen_positions) == top_m:
                    break

        # Fallback: if not enough candidates under capacity, fill remaining slots
        # using least-loaded candidates. This avoids dropping the source.
        if len(chosen_positions) < top_m:
            n_fallback_sources += 1
            already = set(chosen_positions)

            remaining = [int(pos) for pos in cand_order if int(pos) not in already]
            while len(chosen_positions) < top_m and len(remaining) > 0:
                candidates = np.array([int(plan_idx[i, pos]) for pos in remaining], dtype=np.int32)
                counts = target_count[candidates]
                min_count = counts.min()
                feasible_local = np.where(counts == min_count)[0]

                if knn_cost is not None:
                    best_local = feasible_local[
                        np.argmin([knn_cost[i, remaining[q]] for q in feasible_local])
                    ]
                else:
                    best_local = feasible_local[
                        np.argmax([plan_vals[i, remaining[q]] for q in feasible_local])
                    ]

                pos = remaining.pop(int(best_local))
                tgt = int(plan_idx[i, pos])
                chosen_positions.append(int(pos))
                target_count[tgt] += 1

                if target_count[tgt] > max_sources_per_target:
                    n_overflow_assignments += 1

        if len(chosen_positions) < top_m:
            n_underfilled_sources += 1

        chosen_positions = chosen_positions[:top_m]
        chosen_targets = plan_idx[i, chosen_positions].astype(np.int32)
        vals = plan_vals[i, chosen_positions].astype(np.float64)

        vals_sum = vals.sum()
        if vals_sum <= eps:
            # Extremely unlikely unless all plan values are numerically zero.
            vals = np.ones(len(chosen_positions), dtype=np.float64) / len(chosen_positions)
        else:
            vals = vals / vals_sum

        selected_idx[i, :len(chosen_positions)] = chosen_targets
        selected_vals[i, :len(chosen_positions)] = vals.astype(np.float32)

        transported[i] = (
            vals[:, None].astype(np.float32)
            * target_weights[chosen_targets]
        ).sum(axis=0)

    summary = {
        "top_m": int(top_m),
        "max_sources_per_target": int(max_sources_per_target),
        "map_count_min": int(target_count.min()),
        "map_count_max": int(target_count.max()),
        "map_count_mean": float(target_count.mean()),
        "map_count_std": float(target_count.std()),
        "n_targets_with_0_sources": int(np.sum(target_count == 0)),
        "n_targets_with_1_source": int(np.sum(target_count == 1)),
        "n_targets_with_2_sources": int(np.sum(target_count == 2)),
        "n_targets_with_3_sources": int(np.sum(target_count == 3)),
        "n_targets_with_gt_capacity": int(np.sum(target_count > max_sources_per_target)),
        "n_targets_over_capacity": int(np.sum(target_count > max_sources_per_target)),
        "n_fallback_sources": int(n_fallback_sources),
        "n_underfilled_sources": int(n_underfilled_sources),
        "n_overflow_assignments": int(n_overflow_assignments),
    }

    return transported, selected_idx, summary
# ============================================================
# Channel-wise balanced OT map
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
    max_sources_per_target: int,
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
        "min_indegree_required": 2,
    }
    if enforce_target_min_indegree:
        knn_idx, knn_cost, coverage_info = ensure_target_min_indegree(
            f_source=f1,
            f_target=f2,
            knn_idx=knn_idx,
            knn_cost=knn_cost,
            min_indegree=max(1, int(np.ceil(source_to_target_ratio))),
            x_block=x_block,
            y_block=y_block,
            device=device,
        )

    positive = knn_cost[knn_cost > 0]
    if positive.size == 0:
        raise ValueError("All kNN costs are zero.")

    cost_scale = np.median(positive)
    knn_cost_scaled = knn_cost / max(float(cost_scale), 1e-12)

    # Balanced masses: total mass is 1 on both sides.
    # Source and target sample counts do not need to match. If N > M,
    # each target simply carries larger empirical mass than each source.
    a = np.ones(N, dtype=np.float64) / float(N)
    b = np.ones(M, dtype=np.float64) / float(M)

    plan_vals, sinkhorn_diag = sparse_balanced_sinkhorn_knn(
        knn_idx=knn_idx,
        knn_cost=knn_cost_scaled,
        n_target=M,
        reg=reg,
        n_iter=sinkhorn_iter,
        tol=sinkhorn_tol,
        a=a,
        b=b,
    )

    # mapped_idx, map_count_summary = map_from_sparse_plan_capacity_constrained(
    #     plan_idx=knn_idx,
    #     plan_vals=plan_vals,
    #     knn_cost=knn_cost_scaled,
    #     n_target=M,
    #     max_sources_per_target=max_sources_per_target,
    # )
    # transported_weights = w2_raw[mapped_idx]
    transported_weights, selected_idx, map_count_summary = (
        weighted_topm_from_sparse_plan_capacity_constrained(
            plan_idx=knn_idx,
            plan_vals=plan_vals,
            target_weights=w2_raw,
            knn_cost=knn_cost_scaled,
            n_target=M,
            top_m=top_m,
            max_sources_per_target=max_sources_per_target,
        )
    )
    usage = compute_target_usage(knn_idx, plan_vals, n_target=M)
    usage_summary = summarize_usage(usage)

    print(
        f"[channel {channel}] balanced map | "
        f"cost median scaled={np.median(knn_cost_scaled):.6f}, "
        f"row rel linf={sinkhorn_diag['row_rel_linf']:.3e}, "
        f"col rel linf={sinkhorn_diag['col_rel_linf']:.3e}, "
        f"transported std={transported_weights.std():.6f}, "
        f"map count max={map_count_summary['map_count_max']}, "
        f"targets with 2/3 sources="
        f"{map_count_summary['n_targets_with_2_sources']}/"
        f"{map_count_summary['n_targets_with_3_sources']}/{M}, "
        f"overflow targets={map_count_summary['n_targets_over_capacity']}"
    )

    # return {
    #     "channel": channel,
    #     "transported_weights": transported_weights,
    #     "mapped_idx": mapped_idx,
    #     "knn_idx": knn_idx,
    #     "knn_cost_scaled": knn_cost_scaled.astype(np.float32),
    #     "plan_vals": plan_vals,
    #     "usage": usage,
    #     "residual_stats": residual_stats,
    #     "condition_stats": condition_stats,
    #     "usage_summary": usage_summary,
    #     "sinkhorn_diag": sinkhorn_diag,
    #     "coverage_info": coverage_info,
    #     "map_count_summary": map_count_summary,
    #     "source_to_target_ratio": source_to_target_ratio,
    #     "cost_scale": float(cost_scale),
    # }
    return {
        "channel": channel,
        "transported_weights": transported_weights,
        "selected_idx": selected_idx,
        "knn_idx": knn_idx,
        "knn_cost_scaled": knn_cost_scaled.astype(np.float32),
        "plan_vals": plan_vals,
        "usage": usage,
        "residual_stats": residual_stats,
        "condition_stats": condition_stats,
        "usage_summary": usage_summary,
        "sinkhorn_diag": sinkhorn_diag,
        "coverage_info": coverage_info,
        "map_count_summary": map_count_summary,
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
        max_sources_per_target=args["max_sources_per_target"],
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
    knn_k: int = 3,
    top_m: int = 3,
    reg: float = 0.1,
    sinkhorn_iter: int = 1000,
    sinkhorn_tol: float = 1e-8,
    gpu_ids: Tuple[int, ...] = (0, 1, 2, 3),
    x_block: int = 512,
    y_block: int = 2048,
    save_sparse_plan: bool = True,
    use_cond_modes_per_channel: Optional[Dict[int, int]] = None,
    enforce_target_min_indegree: bool = True,
    max_sources_per_target: int = 3,
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
    print(f"source/target sample ratio: {source_to_target_ratio:.6f}")

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
            "reg": reg,
            "sinkhorn_iter": sinkhorn_iter,
            "sinkhorn_tol": sinkhorn_tol,
            "x_block": x_block,
            "y_block": y_block,
            "use_cond_modes_per_channel": use_cond_modes_per_channel,
            "enforce_target_min_indegree": enforce_target_min_indegree,
            "max_sources_per_target": max_sources_per_target,
        })

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(gpu_ids)) as pool:
        print(f"running {len(job_args)} channels on {len(gpu_ids)} GPUs")
        results = pool.map(_channel_worker, job_args)

    usage_summary_all = {}
    sinkhorn_diag_all = {}
    coverage_info_all = {}
    map_count_summary_all = {}

    for out in results:
        c = out["channel"]
        kc = out["residual_stats"]["Kc"]
        transported_all[:, c, :kc] = out["transported_weights"]

        usage_summary_all[f"channel_{c}"] = out["usage_summary"]
        sinkhorn_diag_all[f"channel_{c}"] = out["sinkhorn_diag"]
        coverage_info_all[f"channel_{c}"] = out["coverage_info"]
        map_count_summary_all[f"channel_{c}"] = out["map_count_summary"]

        if save_sparse_plan:
            np.savez_compressed(
                Path(save_dir) / f"channel_{c}_balanced_sparse_plan.npz",
                knn_idx=out["knn_idx"],
                knn_cost_scaled=out["knn_cost_scaled"],
                plan_vals=out["plan_vals"],
                # mapped_idx=out["mapped_idx"],
                selected_idx=out["selected_idx"],
                usage=out["usage"],
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
        "method": "sparse_balanced_ot_capacity_constrained_map",
        "top_m": top_m,
        "alpha": alpha,
        "lambda_cond": lambda_cond,
        "knn_k": knn_k,
        "reg": reg,
        "sinkhorn_iter": sinkhorn_iter,
        "sinkhorn_tol": sinkhorn_tol,
        "gpu_ids": list(gpu_ids),
        "x_block": x_block,
        "y_block": y_block,
        "enforce_target_min_indegree": enforce_target_min_indegree,
        "max_sources_per_target": max_sources_per_target,
        "source_to_target_ratio": source_to_target_ratio,
        "target_min_indegree": int(np.ceil(source_to_target_ratio)),
        "residual_npz_path": residual_npz_path,
        "condition_npz_path": condition_npz_path,
        "final_nc_path": final_nc_path,
        "usage_summary": usage_summary_all,
        "sinkhorn_diag": sinkhorn_diag_all,
        "coverage_info": coverage_info_all,
        "map_count_summary": map_count_summary_all,
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
    return parser.parse_args()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # residual_npz_path = "/data/experiment_outputs/calibration_purpose/pca_weights_2018-2019_and_2020.npz"
    # condition_npz_path = "/data/experiment_outputs/calibration_purpose/pca_weights_lr_2018-2019_and_2020.npz"
    # path_pca_model = "./pca/pca_all_samples_mode_100.npz"
    # source_nc_path = "/data/experiment_outputs/calibration_purpose/regression_2018-2019_all_samples_trained_on_2018-2019.nc"
    # target_nc_path = "/data/experiment_outputs/calibration_purpose/regression_2020_all_samples_trained_on_2018-2019.nc"
    # model = "optimal_transport"
    # residual_npz_path = "/data/experiment_outputs/calibration_purpose/pca_mini_weights_2018-2019_and_2020.npz"
    # condition_npz_path = "/data/experiment_outputs/calibration_purpose/pca_mini_weights_lr_2018-2019_and_2020.npz"
    # path_pca_model = "./pca/pca_mini_model_85.npz"
    # source_nc_path = "/data/experiment_outputs/calibration_purpose/regression_mini_source_samples_2018-2019.nc"
    # target_nc_path = "/data/experiment_outputs/calibration_purpose/regression_mini_target_samples_2020.nc"
    # model="mini_optimal_transport"
    # --------------------------------------------------------
    # Swinlr HRRR 
    # --------------------------------------------------------
    
    # residual_npz_path = "/data/experiment_outputs/calibration_purpose/hrrr/pca_weights_2018-2019_and_2020.npz"
    # condition_npz_path = "/data/experiment_outputs/calibration_purpose/hrrr/pca_lr_weights_2018-2019_and_2020.npz"
    # path_pca_model = "./pca/pca_swinlr_mode_120.npz"
    # source_nc_path = "/data/experiment_outputs/calibration_purpose/swinlr_2018-2019_all_samples_trained_on_2018-2019.nc"
    # target_nc_path = "/data/experiment_outputs/calibration_purpose/swinlr_2020_all_samples_trained_on_2018-2019.nc"
    # model = "swinlr_optimal_transport"
    # channel_names = ["50u", "50v","75u", "75v","100u", "100v","125u", "125v","150u", "150v",]
    # --------------------------------------------------------
    # BlastNet
    # --------------------------------------------------------
    args = parse_args()
    # input_variable_name = args.level
    # residual_npz_path = f"/data/experiment_outputs/calibration_purpose/blastnet/pca_weights_{input_variable_name}.npz"
    # condition_npz_path =  f"/data/experiment_outputs/calibration_purpose/blastnet/pca_lr_weights_{input_variable_name}.npz"
    # path_pca_model = f"/home/nvidia/projects/calibration/pca/pca_blastnet_{input_variable_name}_300.npz"
    # source_nc_path = f"/data/experiment_outputs/calibration_purpose/blastnet/split_train_{input_variable_name}_regression_only.nc"
    # target_nc_path = f"/data/experiment_outputs/calibration_purpose/blastnet/split_calibration_{input_variable_name}_regression_only.nc"
    # model = f"blastnet_{input_variable_name}_optimal_transport_balanced_plan"
    # channel_names=["RHO", "UX", "UY", "UZ"]
    residual_npz_path = args.residual_npz_path
    condition_npz_path = args.condition_npz_path
    path_pca_model = args.path_pca_model
    source_nc_path = args.source_nc_path
    target_nc_path = args.target_nc_path
    save_dir = args.save_dir
    channel_names = args.channel_names
    

    knn_k = 32
    top_m = 3
    max_sources_per_target = 2
    alpha = 1
    lambda_cond = 0.5
    reg = 0.1
    sinkhorn_iter = 1500
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
        reg=reg,
        sinkhorn_iter=sinkhorn_iter,
        sinkhorn_tol=1e-8,
        gpu_ids=(0, 1, 2, 3),
        x_block=512,
        y_block=2048,
        save_sparse_plan=True,
        enforce_target_min_indegree=True,
        max_sources_per_target=max_sources_per_target,
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
    x_res_ot = x_res_ot.squeeze(axis=0) # (1, C, T, H, W) -> (C, T, H, W)

    res_mean = np.mean(x_res_source, axis=(1, 2, 3))      # (C,)
    res_std = np.std(x_res_source, axis=(1, 2, 3))        # (C,)

    res_ot_mean = np.mean(x_res_ot, axis=(1, 2, 3))       # (C,)
    res_ot_std = np.std(x_res_ot, axis=(1, 2, 3))         # (C,)
    print(f"res_ot.shape: {x_res_ot.shape}")
    print(f"res_source.shape: {x_res_source.shape}")
    summary = {
        "res_mean": np.round(res_mean, 3).tolist(),
        "res_ot_mean": np.round(res_ot_mean, 3).tolist(),
        "res_std": np.round(res_std, 3).tolist(),
        "res_ot_std": np.round(res_ot_std, 3).tolist(),
    }


    # plot_image_path = (
    #     f"./plots/{model}/param_sweep/"
    #     f"knn{knn_k}_topm{top_m}_max_sources_per_target{max_sources_per_target}_alpha{alpha}_lambda{lambda_cond}_reg{reg}_sinkhorn_iter{sinkhorn_iter}"
    # )
    # os.makedirs(save_dir, exist_ok=True)

    # with open(f"{save_dir}/residual_summary.json", "w") as f:
    #     json.dump(summary, f, indent=4)
    # plot_ot_result(res_ot=x_res_ot, res_original=x_res_source, save_file=save_dir,
    # # plot_ot_result(res_ot=x_res_target, res_original=x_res_source, save_file=plot_image_path,
    # channel_names=channel_names,channel_idx=[0,1,2,3],idx=[0,10,20,30,40,50]
    # )
