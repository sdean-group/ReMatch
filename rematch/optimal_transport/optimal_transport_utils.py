import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp

import numpy as np
import torch
import xarray as xr
from .fit_pca import load_pca_model
from .utils import save_ot_dataset_as_nc
from .fit_pca import project_weights_to_pixel_space
# ============================================================
# Basic utils
# ============================================================
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ds_to_array(ds: xr.Dataset, name: str = "variable") -> xr.DataArray:
    if len(ds.data_vars) == 0:
        raise ValueError("Dataset has no data variables.")
    return ds.to_array(dim=name)


def _validate_single(data: np.ndarray) -> None:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array.")
    if data.ndim != 4:
        raise ValueError(
            f"Expected shape (C, T, H, W), but got ndim={data.ndim}, shape={data.shape}"
        )


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q05": float(np.percentile(values, 5)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "q95": float(np.percentile(values, 95)),
    }

def compute_pairwise_norm_distribution_sampled(
    data: np.ndarray,
    num_pairs: Optional[int] = None,
    norm_type: str = "l2",
    normalize: bool = True,
    per_channel: bool = False,
    eps: float = 1e-12,
    seed: int = 0,
    dtype=np.float32,
) -> Dict[str, np.ndarray]:
    """
    Approximate within-set pairwise norm distribution by random pair sampling.

    data: (C, T, H, W)
    """

    _validate_single(data)
    C, T, H, W = data.shape
    rng = np.random.default_rng(seed)
    if num_pairs is None:
        num_pairs = T
    data = data.astype(dtype, copy=False)

    if per_channel:
        pairwise_vals = np.empty((num_pairs, C), dtype=np.float64)
    else:
        pairwise_vals = np.empty(num_pairs, dtype=np.float64)

    count = 0
    while count < num_pairs:
        i = rng.integers(0, T)
        j = rng.integers(0, T)
        if i == j:
            continue

        zi = data[:, i]
        zj = data[:, j]
        diff = zi - zj

        if per_channel:
            if norm_type == "l2":
                vals = np.sqrt(np.sum(diff**2, axis=(-2, -1)))
                if normalize:
                    vals = vals / np.sqrt(H * W + eps)
            elif norm_type == "l1":
                vals = np.sum(np.abs(diff), axis=(-2, -1))
                if normalize:
                    vals = vals / (H * W + eps)
            elif norm_type == "mse":
                vals = np.mean(diff**2, axis=(-2, -1))
            elif norm_type == "rmse":
                vals = np.sqrt(np.mean(diff**2, axis=(-2, -1)))
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}")

            pairwise_vals[count] = vals

        else:
            if norm_type == "l2":
                val = np.sqrt(np.sum(diff**2))
                if normalize:
                    val = val / np.sqrt(C * H * W + eps)
            elif norm_type == "l1":
                val = np.sum(np.abs(diff))
                if normalize:
                    val = val / (C * H * W + eps)
            elif norm_type == "mse":
                val = np.mean(diff**2)
            elif norm_type == "rmse":
                val = np.sqrt(np.mean(diff**2))
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}")

            pairwise_vals[count] = val

        count += 1

    flat = pairwise_vals.reshape(-1)
    out = {
        "pairwise_values": pairwise_vals,
        "num_pairs": num_pairs,
    }
    out.update(_summary_stats(flat))
    return out

# ============================================================
# NetCDF loader compatible with your existing structure
# ============================================================
def load_training_data(nc_path: str):
    truth_ds = xr.open_dataset(nc_path, group="truth")
    pred_ds  = xr.open_dataset(nc_path, group="prediction").isel(ensemble=0, drop=True)
    inp_ds   = xr.open_dataset(nc_path, group="input")

    res_ds = truth_ds - pred_ds
    gt_arr  = ds_to_array(truth_ds)   # (C, T, H, W)
    reg_arr = ds_to_array(pred_ds)    # (C, T, H, W)
    res_arr = ds_to_array(res_ds)     # (C, T, H, W)
    inp_arr = ds_to_array(inp_ds)     # (C_in, T, H, W)
    C,T,H,W = gt_arr.shape
    
    # inp_arr = inp_arr[:C, :, :, :]
    return np.array(inp_arr), np.array(inp_arr[:C, :, :, :]), np.array(res_arr), np.array(gt_arr)


# ============================================================
# Weight dataset loader
# ============================================================
def load_weight_pair_dataset2(filepath: str):
    dataset = np.load(filepath, allow_pickle=True)
    gt_weights_source = dataset["gt_weights_source"]   # source
    gt_weights_target   = dataset["gt_weights_target"]     # target
    mode_mask = dataset["mode_mask"]
    n_modes_per_channel = dataset["n_modes_per_channel"]
    return gt_weights_source, gt_weights_target, mode_mask, n_modes_per_channel


# ============================================================
# Feature prep
# ============================================================
def flatten_condition_features(
    cond_source: np.ndarray,   # (T1, C, Kmax)
    cond_target: np.ndarray,   # (T2, C, Kmax)
    n_modes_per_channel: np.ndarray,
    use_modes_per_channel: Optional[Dict[int, int]] = None,
    stats_from_target: bool = True,
    eps: float = 1e-6,
):
    T1, C, Kmax = cond_source.shape
    T2, C2, Kmax2 = cond_target.shape
    assert C == C2 and Kmax == Kmax2

    parts1 = []
    parts2 = []

    for c in range(C):
        kc = int(n_modes_per_channel[c])
        if use_modes_per_channel is not None and c in use_modes_per_channel:
            kc = min(kc, int(use_modes_per_channel[c]))
        parts1.append(cond_source[:, c, :kc])
        parts2.append(cond_target[:, c, :kc])

    z1 = np.concatenate(parts1, axis=1).astype(np.float32)
    z2 = np.concatenate(parts2, axis=1).astype(np.float32)

    ref = z2 if stats_from_target else np.concatenate([z1, z2], axis=0)
    mu = ref.mean(axis=0, keepdims=True)
    std = ref.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)

    z1 = (z1 - mu) / std
    z2 = (z2 - mu) / std

    return z1, z2, {"mu": mu.astype(np.float32), "std": std.astype(np.float32)}

def get_channel_weights(
    weights_source,
    weights_target,
    channel,
    n_modes_per_channel,
    normalize_from_target=True,
    eps=1e-6,
):
    kc = int(n_modes_per_channel[channel])

    w1_raw = weights_source[:, channel, :kc].astype(np.float32)
    w2_raw = weights_target[:, channel, :kc].astype(np.float32)

    ref = w2_raw if normalize_from_target else np.concatenate([w1_raw, w2_raw], axis=0)
    mu = ref.mean(axis=0, keepdims=True)
    std = np.maximum(ref.std(axis=0, keepdims=True), eps)

    w1_norm = (w1_raw - mu) / std
    w2_norm = (w2_raw - mu) / std

    return w1_raw, w2_raw, w1_norm, w2_norm, {
        "mu": mu.astype(np.float32),
        "std": std.astype(np.float32),
        "Kc": kc,
    }

def build_augmented_features(
    w1c: np.ndarray,
    w2c: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    alpha: float = 1.0,
    lambda_cond: float = 1.0,
):
    a = np.float32(np.sqrt(alpha))
    l = np.float32(np.sqrt(lambda_cond))
    f1 = np.concatenate([a * w1c, l * z1], axis=1).astype(np.float32)
    f2 = np.concatenate([a * w2c, l * z2], axis=1).astype(np.float32)
    return f1, f2

def blockwise_topk_sqdist(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 128,
    x_block: int = 512,
    y_block: int = 2048,
    device: str = "cuda",
):
    N, D = X.shape
    M, D2 = Y.shape
    assert D == D2
    assert k <= M

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    Y_t = torch.from_numpy(Y).to(dev, non_blocking=True)
    y_norm = (Y_t * Y_t).sum(dim=1)

    knn_idx_all = np.empty((N, k), dtype=np.int32)
    knn_dist_all = np.empty((N, k), dtype=np.float32)

    for i0 in range(0, N, x_block):
        i1 = min(i0 + x_block, N)
        X_blk = torch.from_numpy(X[i0:i1]).to(dev, non_blocking=True)
        x_norm = (X_blk * X_blk).sum(dim=1, keepdim=True)

        best_dist = None
        best_idx = None

        for j0 in range(0, M, y_block):
            j1 = min(j0 + y_block, M)
            Y_blk = Y_t[j0:j1]

            # squared L2
            dist = x_norm + y_norm[j0:j1].unsqueeze(0) - 2.0 * (X_blk @ Y_blk.T)
            dist = torch.clamp(dist, min=0.0)

            # mean squared distance
            dist = dist / float(D)

            d_part, i_part = torch.topk(
                dist, k=min(k, dist.shape[1]), dim=1, largest=False
            )
            i_part = i_part + j0

            if best_dist is None:
                best_dist = d_part
                best_idx = i_part
            else:
                cat_dist = torch.cat([best_dist, d_part], dim=1)
                cat_idx = torch.cat([best_idx, i_part], dim=1)
                d_new, pos = torch.topk(cat_dist, k=k, dim=1, largest=False)
                i_new = torch.gather(cat_idx, 1, pos)
                best_dist, best_idx = d_new, i_new

        knn_idx_all[i0:i1] = best_idx.cpu().numpy().astype(np.int32)
        knn_dist_all[i0:i1] = best_dist.cpu().numpy().astype(np.float32)

        del X_blk, x_norm, best_dist, best_idx
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return knn_idx_all, knn_dist_all

# ============================================================
# Sparse unbalanced Sinkhorn
# ============================================================
def sparse_unbalanced_sinkhorn_knn(
    knn_idx: np.ndarray,
    knn_cost: np.ndarray,
    n_target: int,
    reg: float = 0.05,
    reg_m: float = 5.0,
    n_iter: int = 300,
    tol: float = 1e-6,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    eps: float = 1e-12,
):
    N, k = knn_idx.shape

    if a is None:
        a = np.ones(N, dtype=np.float64) / N
    if b is None:
        b = np.ones(n_target, dtype=np.float64) / n_target

    K = np.exp(-knn_cost.astype(np.float64) / reg)
    tau = reg_m / (reg_m + reg)

    u = np.ones(N, dtype=np.float64)
    v = np.ones(n_target, dtype=np.float64)
    prev_u = u.copy()

    for _ in range(n_iter):
        Kv = (K * v[knn_idx]).sum(axis=1) + eps
        u = (a / Kv) ** tau

        KTu = np.zeros(n_target, dtype=np.float64)
        np.add.at(KTu, knn_idx.reshape(-1), (K * u[:, None]).reshape(-1))
        KTu = KTu + eps
        v = (b / KTu) ** tau

        rel = np.max(np.abs(u - prev_u) / (np.abs(prev_u) + eps))
        prev_u = u.copy()
        if rel < tol:
            break

    plan_vals = (u[:, None] * K) * v[knn_idx]
    row_sum = plan_vals.sum(axis=1) + eps
    return plan_vals.astype(np.float32), row_sum.astype(np.float32)


def barycentric_from_sparse_plan(
    plan_idx: np.ndarray,
    plan_vals: np.ndarray,
    target_weights: np.ndarray,
):
    N, k = plan_idx.shape
    Kc = target_weights.shape[1]

    w_norm = plan_vals / (plan_vals.sum(axis=1, keepdims=True) + 1e-12)
    transported = np.zeros((N, Kc), dtype=np.float32)

    for i in range(N):
        transported[i] = (w_norm[i, :, None] * target_weights[plan_idx[i]]).sum(axis=0)
    return transported

def solve_one_channel_sparse_conditional_ot(
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
    reg: float,
    reg_m: float,
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

    knn_idx, knn_cost = blockwise_topk_sqdist(
        X=f1,
        Y=f2,
        k=knn_k,
        x_block=x_block,
        y_block=y_block,
        device=device,
    )

    # robust cost scaling
    positive = knn_cost[knn_cost > 0]
    if positive.size == 0:
        raise ValueError("All kNN costs are zero.")

    cost_scale = np.median(positive)
    knn_cost_scaled = knn_cost / max(cost_scale, 1e-12)

    plan_vals, row_sum = sparse_unbalanced_sinkhorn_knn(
        knn_idx=knn_idx,
        knn_cost=knn_cost_scaled,
        n_target=w2_norm.shape[0],
        reg=reg,          # 이제 reg는 scaled cost 기준
        reg_m=reg_m,
        n_iter=sinkhorn_iter,
        tol=sinkhorn_tol,
    )

    if knn_k == 1:
        transported_weights = w2_raw[knn_idx[:, 0]]
    else:
        transported_weights = barycentric_from_sparse_plan(
            plan_idx=knn_idx,
            plan_vals=plan_vals,
            target_weights=w2_raw,
        )

    chosen_idx = knn_idx[:, 0]
    print(f"[channel {channel}] cost min/max/median: "
          f"{knn_cost.min():.6f} / {knn_cost.max():.6f} / {np.median(knn_cost):.6f}")
    print(f"[channel {channel}] scaled cost min/max/median: "
          f"{knn_cost_scaled.min():.6f} / {knn_cost_scaled.max():.6f} / {np.median(knn_cost_scaled):.6f}")
    print(f"[channel {channel}] raw target std: {w2_raw.std():.6f}")
    print(f"[channel {channel}] transported std: {transported_weights.std():.6f}")
    print(f"[channel {channel}] max diff to chosen raw target: "
          f"{np.abs(transported_weights - w2_raw[chosen_idx]).max():.6f}")
    print(f"[channel {channel}] nonzero fraction in plan_vals: {np.mean(plan_vals > 0):.6f}")

    return {
        "channel": channel,
        "transported_weights": transported_weights,
        "knn_idx": knn_idx,
        "plan_vals": plan_vals,
        "residual_stats": residual_stats,
        "condition_stats": condition_stats,
    }

def _channel_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["local_gpu"])
    torch.cuda.set_device(0)

    out = solve_one_channel_sparse_conditional_ot(
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
        reg=args["reg"],
        reg_m=args["reg_m"],
        sinkhorn_iter=args["sinkhorn_iter"],
        sinkhorn_tol=args["sinkhorn_tol"],
        device="cuda",
        x_block=args["x_block"],
        y_block=args["y_block"],
        use_cond_modes_per_channel=args["use_cond_modes_per_channel"],
    )
    return out



# ============================================================
# Full multi-GPU pipeline
# ============================================================
def run_channelwise_sparse_conditional_ot_multigpu(
    residual_npz_path: str,
    condition_npz_path: str,
    pca_model: Dict[int, Dict[str, np.ndarray]],
    x_lr: np.ndarray,     # (C, T1, H, W)
    x_lr_full: np.ndarray,     # (C_in, T1, H, W)
    x_gt: np.ndarray,     # (C, T1, H, W)
    save_dir: str,
    final_nc_path: str,
    channel_names: List[str],
    alpha: float = 1.0,
    lambda_cond: float = 0.5,
    knn_k: int = 128,
    reg: float = 0.05,
    reg_m: float = 5.0,
    sinkhorn_iter: int = 300,
    sinkhorn_tol: float = 1e-6,
    gpu_ids: Tuple[int, ...] = (0, 1, 2, 3),
    x_block: int = 512,
    y_block: int = 2048,
    save_sparse_plan: bool = True,
    use_cond_modes_per_channel: Optional[Dict[int, int]] = None,
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

    T1, C, Kmax_res = res_source.shape
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
            "reg": reg,
            "reg_m": reg_m,
            "sinkhorn_iter": sinkhorn_iter,
            "sinkhorn_tol": sinkhorn_tol,
            "x_block": x_block,
            "y_block": y_block,
            "use_cond_modes_per_channel": use_cond_modes_per_channel,
        })

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(gpu_ids)) as pool:
        print(f"running {len(gpu_ids)} channels on {len(gpu_ids)} GPUs")
        results = pool.map(_channel_worker, job_args)

    for out in results:
        c = out["channel"]
        kc = out["residual_stats"]["Kc"]
        transported_all[:, c, :kc] = out["transported_weights"]

        if save_sparse_plan:
            np.savez_compressed(
                Path(save_dir) / f"channel_{c}_sparse_plan.npz",
                knn_idx=out["knn_idx"],
                plan_vals=out["plan_vals"],
                channel=np.array(c, dtype=np.int32),
                Kc=np.array(kc, dtype=np.int32),
            )

    np.save(Path(save_dir) / "transported_source_residual_weights.npy", transported_all)

    # inverse PCA -> residual_ot
    # print(pca_model)
    x_res_ot = project_weights_to_pixel_space(
        weights=transported_all,
        pca_model=pca_model,
        n_modes_per_channel=res_n_modes,
    )  # (C, T1, H, W)

    np.save(Path(save_dir) / "source_residual_ot_reconstructed.npy", x_res_ot)



    meta = {
        "alpha": alpha,
        "lambda_cond": lambda_cond,
        "knn_k": knn_k,
        "reg": reg,
        "reg_m": reg_m,
        "sinkhorn_iter": sinkhorn_iter,
        "sinkhorn_tol": sinkhorn_tol,
        "gpu_ids": list(gpu_ids),
        "x_block": x_block,
        "y_block": y_block,
        "residual_npz_path": residual_npz_path,
        "condition_npz_path": condition_npz_path,
        "final_nc_path": final_nc_path,
    }
    with open(Path(save_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return transported_all, x_res_ot
