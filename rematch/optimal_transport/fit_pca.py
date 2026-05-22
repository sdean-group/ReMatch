#pca.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np
from sklearn.utils.extmath import randomized_svd
import argparse
from .utils import load_training_data, load_validation_data
import os 
from .utils import plot_pca_validation
@dataclass
class ChannelPCAModel:
    channel_index: int
    mean: np.ndarray                          # (W*H,)
    components: np.ndarray                    # (K, W*H)
    singular_values: np.ndarray               # (K,)
    explained_variance: np.ndarray            # (K,)
    explained_variance_ratio: np.ndarray      # (K,) relative to TOTAL variance
    spatial_shape: tuple[int, int]            # (W, H)
    n_train_samples: int                      # T
    n_features: int                           # W*H
    n_components: int                         # K
    center: bool

    # extra summaries
    total_variance: float
    captured_variance_ratio_topk: float
    effective_rank_topk: float


@dataclass
class PCAModel:
    input_shape: tuple[int, int, int, int]    # (C, T, W, H)
    per_channel: Dict[int, ChannelPCAModel]


def _validate_input(x: np.ndarray) -> None:
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy.ndarray.")
    if x.ndim != 4:
        raise ValueError(f"Expected shape (C, T, W, H), got {x.shape}.")
    C, T, W, H = x.shape
    if C < 1 or T < 2 or W < 1 or H < 1:
        raise ValueError(f"Invalid shape {x.shape}.")


def fit_truncated_pca_per_channel(
    x: np.ndarray,
    n_components: int = 100,
    center: bool = True,
    n_iter: int = 5,
    random_state: int = 0,
    save_dtype: np.dtype = np.float32,
) -> PCAModel:
    """
    Fit top-k truncated PCA independently for each channel.

    Parameters
    ----------
    x : np.ndarray
        Shape (C, T, W, H)
    n_components : int
        Number of PCA modes to compute/store per channel.
    center : bool
        Whether to subtract feature-wise temporal mean.
    n_iter : int
        Randomized SVD power iterations.
    random_state : int
        Random seed for randomized SVD.
    save_dtype : np.dtype
        dtype used for stored arrays (typically float32 to reduce size).

    Returns
    -------
    PCAModel
    """
    _validate_input(x)
    C, T, W, H = x.shape
    F = W * H
    max_rank = min(T, F)

    if not (1 <= n_components <= max_rank):
        raise ValueError(
            f"n_components must be in [1, {max_rank}], got {n_components}."
        )

    per_channel: Dict[int, ChannelPCAModel] = {}

    for c in range(C):
        X = x[c].reshape(T, F).astype(np.float64, copy=False)

        if center:
            mean = X.mean(axis=0)
            Xc = X - mean[None, :]
        else:
            mean = np.zeros(F, dtype=np.float64)
            Xc = X

        # Exact total variance without full SVD
        total_variance = float(np.sum(Xc * Xc) / (T - 1))

        # randomized truncated SVD
        U, S, Vt = randomized_svd(
            Xc,
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state,
        )

        explained_variance = (S ** 2) / (T - 1)
        explained_variance_ratio = (
            explained_variance / total_variance
            if total_variance > 0
            else np.zeros_like(explained_variance)
        )

        captured_variance_ratio_topk = float(np.sum(explained_variance_ratio))

        # effective rank computed only over top-k spectrum
        denom = np.sum(explained_variance ** 2)
        if denom > 0:
            effective_rank_topk = float((np.sum(explained_variance) ** 2) / denom)
        else:
            effective_rank_topk = 0.0

        per_channel[c] = ChannelPCAModel(
            channel_index=c,
            mean=mean.astype(save_dtype, copy=False),
            components=Vt.astype(save_dtype, copy=False),
            singular_values=S.astype(save_dtype, copy=False),
            explained_variance=explained_variance.astype(save_dtype, copy=False),
            explained_variance_ratio=explained_variance_ratio.astype(save_dtype, copy=False),
            spatial_shape=(W, H),
            n_train_samples=T,
            n_features=F,
            n_components=n_components,
            center=center,
            total_variance=total_variance,
            captured_variance_ratio_topk=captured_variance_ratio_topk,
            effective_rank_topk=effective_rank_topk,
        )

        print(
            f"[Channel {c}] captured_variance_ratio_top{n_components}="
            f"{captured_variance_ratio_topk:.6f}, "
            f"effective_rank_top{n_components}={effective_rank_topk:.3f}"
        )

    return PCAModel(
        input_shape=(C, T, W, H),
        per_channel=per_channel,
    )

# =========================================================
# Projection
# =========================================================
def project_to_pca_weights_per_channel(
    x: np.ndarray,
    pca_model: PCAModel,
    out_dtype: np.dtype = np.float32,
) -> Dict[int, np.ndarray]:
    """
    Project x onto already-fitted PCA bases.

    Parameters
    ----------
    x : np.ndarray
        Shape (C, T, W, H) or (S, C, T, W, H)

    Returns
    -------
    weights_dict : Dict[int, np.ndarray]
        If input is (C, T, W, H):
            weights_dict[c] has shape (T, K)
        If input is (S, C, T, W, H):
            weights_dict[c] has shape (S, T, K)
    """
    if x.ndim != 4 and x.ndim != 5:
        raise ValueError(f"x must have shape (C, T, W, H) or (S, C, T, W, H), got {x.shape}")

    if x.ndim == 4:
        x = x[None, ...]   # -> (1, C, T, W, H)

    S, C, T, W, H = x.shape

    if C != len(pca_model.per_channel):
        raise ValueError(
            f"Channel mismatch: data has C={C}, PCA model has {len(pca_model.per_channel)}"
        )

    weights_dict: Dict[int, np.ndarray] = {}

    for c in range(C):
        ch = pca_model.per_channel[c]
        Wm, Hm = ch.spatial_shape
        if (W, H) != (Wm, Hm):
            raise ValueError(
                f"Spatial mismatch at channel {c}: data {(W, H)} vs model {(Wm, Hm)}"
            )

        Vt = ch.components.astype(np.float64, copy=False)   # (K, F)

        proj_list = []
        for s in range(S):
            X = x[s, c].reshape(T, W * H).astype(np.float64, copy=False)  # (T, F)

            if ch.center:
                Xc = X - ch.mean[None, :]
            else:
                Xc = X

            W_proj = Xc @ Vt.T   # (T, K)
            proj_list.append(W_proj.astype(out_dtype, copy=False))

        proj_arr = np.stack(proj_list, axis=0)  # (S, T, K)

        if S == 1:
            weights_dict[c] = proj_arr[0]   # (T, K)
        else:
            weights_dict[c] = proj_arr      # (S, T, K)

    return weights_dict
def save_pca_model(model: PCAModel, filepath: str) -> None:

    save_dict: Dict[str, Any] = {}
    C, T, W, H = model.input_shape

    save_dict["meta_input_shape"] = np.array([C, T, W, H], dtype=np.int64)
    save_dict["meta_num_channels"] = np.array([len(model.per_channel)], dtype=np.int64)

    for c, ch in model.per_channel.items():
        prefix = f"ch{c}_"
        save_dict[prefix + "channel_index"] = np.array([ch.channel_index], dtype=np.int64)
        save_dict[prefix + "mean"] = ch.mean
        save_dict[prefix + "components"] = ch.components
        save_dict[prefix + "singular_values"] = ch.singular_values
        save_dict[prefix + "explained_variance"] = ch.explained_variance
        save_dict[prefix + "explained_variance_ratio"] = ch.explained_variance_ratio
        save_dict[prefix + "spatial_shape"] = np.array(ch.spatial_shape, dtype=np.int64)
        save_dict[prefix + "n_train_samples"] = np.array([ch.n_train_samples], dtype=np.int64)
        save_dict[prefix + "n_features"] = np.array([ch.n_features], dtype=np.int64)
        save_dict[prefix + "n_components"] = np.array([ch.n_components], dtype=np.int64)
        save_dict[prefix + "center"] = np.array([int(ch.center)], dtype=np.int64)

        save_dict[prefix + "total_variance"] = np.array([ch.total_variance], dtype=np.float64)
        save_dict[prefix + "captured_variance_ratio_topk"] = np.array(
            [ch.captured_variance_ratio_topk], dtype=np.float64
        )
        save_dict[prefix + "effective_rank_topk"] = np.array(
            [ch.effective_rank_topk], dtype=np.float64
        )

    np.savez_compressed(filepath, **save_dict)

def save_weight_pair_dataset(
    filepath: str,
    gt_weights: np.ndarray,      # (T, C, Kmax)
    pred_weights: np.ndarray,    # (S, T, C, Kmax)
    inp_weights: np.ndarray,     # (T, C, Kmax)
    reg_weights: np.ndarray,     # (T, C, Kmax)
    mode_mask: np.ndarray,       # (C, Kmax)
    n_modes_per_channel: Dict[int, int],
) -> None:
    """
    Save dataset to npz.
    """
    C = len(n_modes_per_channel)
    n_modes_arr = np.zeros((C,), dtype=np.int32)
    for c in range(C):
        n_modes_arr[c] = int(n_modes_per_channel[c])

    np.savez_compressed(
        filepath,
        gt_weights=gt_weights,
        pred_weights=pred_weights,
        inp_weights=inp_weights,
        reg_weights=reg_weights,
        mode_mask=mode_mask,
        n_modes_per_channel=n_modes_arr,
    )
def save_weight_pair_dataset2(
    filepath: str,
    gt_weights_source: np.ndarray,      # (T, C, Kmax)
    gt_weights_target: np.ndarray,      # (T, C, Kmax)
    mode_mask: np.ndarray,       # (C, Kmax)
    n_modes_per_channel: Dict[int, int],
) -> None:
    """
    Save dataset to npz.
    """
    C = len(n_modes_per_channel)
    n_modes_arr = np.zeros((C,), dtype=np.int32)
    for c in range(C):
        n_modes_arr[c] = int(n_modes_per_channel[c])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(
        filepath,
        gt_weights_source=gt_weights_source,
        gt_weights_target=gt_weights_target,
        mode_mask=mode_mask,
        n_modes_per_channel=n_modes_arr,
    )

def build_pca_weight_pair_dataset(
    gt_res: np.ndarray,     # (C, T, W, H)
    pred_res: np.ndarray,   # (S, C, T, W, H)
    inp: np.ndarray,        # (C, T, W, H)
    reg: np.ndarray,        # (C, T, W, H)
    model,
    n_modes_per_channel: Optional[Dict[int, int]] = None,
    use_effective_rank: bool = False,
    round_effective_rank: str = "ceil",
    out_dtype=np.float32,
):
    """
    Build dataset of PCA-weight pairs.

    Output:
      gt_weights   : (T, C, Kmax)
      pred_weights : (S, T, C, Kmax)
      mode_mask    : (C, Kmax)
      n_modes_per_channel : dict[channel] = k_c

    For one data point (t_i, c_i):
      gt_vec   = gt_weights[t_i, c_i, :]       # (Kmax,)
      pred_mat = pred_weights[:, t_i, c_i, :]  # (S, Kmax)
    """
    n_modes = resolve_n_modes_per_channel(
        model,
        n_modes_per_channel=n_modes_per_channel,
        use_effective_rank=use_effective_rank,
        round_effective_rank=round_effective_rank,
    )
    print("projecting gt_residual to pca weights")
    gt_weights, mode_mask_1 = project_gt_residual_to_pca_weights(
        gt_res=gt_res,
        model=model,
        n_modes_per_channel=n_modes,
        out_dtype=out_dtype,
    )
    print("projecting pred_residual to pca weights")
    pred_weights, mode_mask_2 = project_pred_residual_to_pca_weights(
        pred_res=pred_res,
        model=model,
        n_modes_per_channel=n_modes,
        out_dtype=out_dtype,
    )
    inp_weights, mode_mask_3 = project_gt_residual_to_pca_weights(
        gt_res=inp,
        model=model,
        n_modes_per_channel=n_modes,
        out_dtype=out_dtype,
    )
    reg_weights, mode_mask_4 = project_gt_residual_to_pca_weights(
        gt_res=reg,
        model=model,
        n_modes_per_channel=n_modes,
        out_dtype=out_dtype,
    )

    if not np.array_equal(mode_mask_1, mode_mask_2):
        raise RuntimeError("mode_mask mismatch between gt and pred projections")

    return {
        "gt_weights": gt_weights,               # (T, C, Kmax)
        "pred_weights": pred_weights,           # (S, T, C, Kmax)
        "inp_weights": inp_weights,             # (T, C, Kmax)
        "reg_weights": reg_weights,             # (T, C, Kmax)
        "mode_mask": mode_mask_1,               # (C, Kmax)
        "n_modes_per_channel": n_modes,
    }
def build_pca_weight_pair_dataset2(
    gt_res_train: np.ndarray,     # (C, T, W, H)
    gt_res_val: np.ndarray,     # (C, T, W, H)
    model,
    n_modes_per_channel: Optional[Dict[int, int]] = None,
    use_effective_rank: bool = False,
    round_effective_rank: str = "ceil",
    out_dtype=np.float32,
):
    """
    Build dataset of PCA-weight pairs.

    Output:
      gt_weights   : (T, C, Kmax)
      pred_weights : (S, T, C, Kmax)
      mode_mask    : (C, Kmax)
      n_modes_per_channel : dict[channel] = k_c

    For one data point (t_i, c_i):
      gt_vec   = gt_weights[t_i, c_i, :]       # (Kmax,)
      pred_mat = pred_weights[:, t_i, c_i, :]  # (S, Kmax)
    """
    n_modes = resolve_n_modes_per_channel(
        model,
        n_modes_per_channel=n_modes_per_channel,
        use_effective_rank=use_effective_rank,
        round_effective_rank=round_effective_rank,
    )
    print("projecting gt_residual to pca weights")
    gt_weights_source, mode_mask = project_gt_residual_to_pca_weights(
        gt_res=gt_res_train,
        model=model,
        n_modes_per_channel=n_modes,
        out_dtype=out_dtype,
    )
    print("projecting pred_residual to pca weights")
    gt_weights_target, mode_mask = project_gt_residual_to_pca_weights(
        gt_res=gt_res_val,
        model=model,
        n_modes_per_channel=n_modes,
        out_dtype=out_dtype,
    )

    return {
        "gt_weights_source": gt_weights_source,               # (T, C, Kmax)
        "gt_weights_target": gt_weights_target,               # (T, C, Kmax)
        "mode_mask": mode_mask,               # (C, Kmax)
        "n_modes_per_channel": n_modes,
    }
    
def resolve_n_modes_per_channel(
    model,
    n_modes_per_channel: Optional[Dict[int, int]] = None,
    use_effective_rank: bool = False,
    round_effective_rank: str = "ceil",
) -> Dict[int, int]:
    """
    Decide how many PCA modes to use for each channel.

    Priority:
    1) n_modes_per_channel explicitly given
    2) use_effective_rank=True -> use stored effective_rank_topk
    3) otherwise use all stored modes

    Returns
    -------
    dict[channel] = k_c
    """
    out = {}

    for c, ch in model.per_channel.items():
        if n_modes_per_channel is not None:
            k = int(n_modes_per_channel[c])
        elif use_effective_rank:
            eff = float(ch.effective_rank_topk)
            if round_effective_rank == "ceil":
                k = int(np.ceil(eff))
            elif round_effective_rank == "floor":
                k = int(np.floor(eff))
            elif round_effective_rank == "round":
                k = int(np.round(eff))
            else:
                raise ValueError(
                    "round_effective_rank must be one of {'ceil','floor','round'}"
                )
        else:
            k = int(ch.n_components)
        if not (1 <= k <= ch.n_components):
            raise ValueError(
                f"Channel {c}: requested k={k}, but stored n_components={ch.n_components}"
            )

        out[c] = k

    return out

def build_mode_mask(
    n_modes_per_channel: Dict[int, int]
) -> np.ndarray:
    """
    Build mask of shape (C, Kmax), where True means valid mode for that channel.
    """
    C = len(n_modes_per_channel)
    Kmax = max(n_modes_per_channel.values())
    mask = np.zeros((C, Kmax), dtype=bool)

    for c, k in n_modes_per_channel.items():
        mask[c, :k] = True

    return mask
def load_pca_model(filepath: str) -> PCAModel:
    data = np.load(filepath, allow_pickle=False)

    input_shape = tuple(int(v) for v in data["meta_input_shape"])
    num_channels = int(data["meta_num_channels"][0])

    per_channel: Dict[int, ChannelPCAModel] = {}

    for c in range(num_channels):
        prefix = f"ch{c}_"
        per_channel[c] = ChannelPCAModel(
            channel_index=int(data[prefix + "channel_index"][0]),
            mean=data[prefix + "mean"],
            components=data[prefix + "components"],
            singular_values=data[prefix + "singular_values"],
            explained_variance=data[prefix + "explained_variance"],
            explained_variance_ratio=data[prefix + "explained_variance_ratio"],
            spatial_shape=tuple(int(v) for v in data[prefix + "spatial_shape"]),
            n_train_samples=int(data[prefix + "n_train_samples"][0]),
            n_features=int(data[prefix + "n_features"][0]),
            n_components=int(data[prefix + "n_components"][0]),
            center=bool(data[prefix + "center"][0]),
            total_variance=float(data[prefix + "total_variance"][0]),
            captured_variance_ratio_topk=float(data[prefix + "captured_variance_ratio_topk"][0]),
            effective_rank_topk=float(data[prefix + "effective_rank_topk"][0]),
        )
    return PCAModel(
        input_shape=input_shape,
        per_channel=per_channel,
    )


def summarize_pca_model(model: PCAModel) -> Dict[int, Dict[str, float]]:
    summary: Dict[int, Dict[str, float]] = {}
    for c, ch in model.per_channel.items():
        summary[c] = {
            "effective_rank_topk": ch.effective_rank_topk,
            "captured_variance_ratio_topk": ch.captured_variance_ratio_topk,
            "n_components_stored": float(ch.n_components),
        }
    return summary


def project_onto_pca(
    x: np.ndarray,
    model: PCAModel,
    n_modes_per_channel: Optional[Dict[int, int]] = None,
    reconstruct: bool = False,
) -> tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
    """
    Project new data x of shape (C, T_new, W, H) onto stored PCA basis.

    Returns
    -------
    scores_dict : dict[channel] = (T_new, k)
    x_recon : np.ndarray or None
        Shape (C, T_new, W, H) if reconstruct=True
    """
    _validate_input(x)
    C, T_new, W, H = x.shape
    C_model, _, Wm, Hm = model.input_shape

    if C != C_model or W != Wm or H != Hm:
        raise ValueError(
            f"Shape mismatch: x={x.shape}, model expects channels/spatial={model.input_shape}"
        )

    scores_dict: Dict[int, np.ndarray] = {}
    x_recon = np.zeros_like(x, dtype=np.float32) if reconstruct else None

    for c in range(C):
        ch = model.per_channel[c]
        X = x[c].reshape(T_new, W * H).astype(np.float64, copy=False)

        if ch.center:
            Xc = X - ch.mean[None, :]
        else:
            Xc = X

        if n_modes_per_channel is None:
            k = ch.n_components
        else:
            k = int(n_modes_per_channel[c])

        if not (1 <= k <= ch.n_components):
            raise ValueError(
                f"Channel {c}: requested k={k}, but stored n_components={ch.n_components}."
            )
        comps_k = ch.components[:k, :].astype(np.float64, copy=False)
        scores = Xc @ comps_k.T
        scores_dict[c] = scores

        if reconstruct:
            Xrec = scores @ comps_k
            if ch.center:
                Xrec = Xrec + ch.mean[None, :]
            x_recon[c] = Xrec.reshape(T_new, W, H).astype(np.float32, copy=False)

    return scores_dict, x_recon
def project_gt_residual_to_pca_weights(
    gt_res: np.ndarray,   # (C, T, W, H)
    model,
    n_modes_per_channel: Dict[int, int],
    out_dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project GT residual onto the train PCA basis.

    Returns
    -------
    gt_weights : np.ndarray
        Shape (T, C, Kmax)
    mode_mask : np.ndarray
        Shape (C, Kmax), bool
    """
    if gt_res.ndim != 4:
        raise ValueError(f"Expected gt_res shape (C,T,W,H), got {gt_res.shape}")

    C, T, W, H = gt_res.shape
    if C != len(model.per_channel):
        raise ValueError(f"Channel mismatch: gt_res has C={C}, model has {len(model.per_channel)}")

    Kmax = max(n_modes_per_channel.values())
    gt_weights = np.zeros((T, C, Kmax), dtype=out_dtype)
    mode_mask = build_mode_mask(n_modes_per_channel)

    for c in range(C):
        ch = model.per_channel[c]
        Wm, Hm = ch.spatial_shape
        if (W, H) != (Wm, Hm):
            raise ValueError(
                f"Channel {c}: spatial mismatch, gt_res={(W,H)}, model={(Wm,Hm)}"
            )

        k = n_modes_per_channel[c]
        comps = ch.components[:k, :].astype(np.float64, copy=False)   # (k, WH)

        X = gt_res[c].reshape(T, W * H).astype(np.float64, copy=False)  # (T, WH)
        if ch.center:
            X = X - ch.mean[None, :]
        
        scores = X @ comps.T   # (T, k)
        gt_weights[:, c, :k] = scores.astype(out_dtype, copy=False)
        x_rec = scores @ comps + ch.mean[None, :]
    return gt_weights, mode_mask


def project_pred_residual_to_pca_weights(
    pred_res: np.ndarray,   # (S, C, T, W, H)
    model,
    n_modes_per_channel: Dict[int, int],
    out_dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project ensemble predicted residual onto the same train PCA basis.

    Returns
    -------
    pred_weights : np.ndarray
        Shape (S, T, C, Kmax)
    mode_mask : np.ndarray
        Shape (C, Kmax), bool
    """
    if pred_res.ndim != 5:
        raise ValueError(f"Expected pred_res shape (S,C,T,W,H), got {pred_res.shape}")

    S, C, T, W, H = pred_res.shape
    if C != len(model.per_channel):
        raise ValueError(f"Channel mismatch: pred_res has C={C}, model has {len(model.per_channel)}")

    Kmax = max(n_modes_per_channel.values())
    pred_weights = np.zeros((S, T, C, Kmax), dtype=out_dtype)
    mode_mask = build_mode_mask(n_modes_per_channel)

    for c in range(C):
        ch = model.per_channel[c]
        Wm, Hm = ch.spatial_shape
        if (W, H) != (Wm, Hm):
            raise ValueError(
                f"Channel {c}: spatial mismatch, pred_res={(W,H)}, model={(Wm,Hm)}"
            )

        k = n_modes_per_channel[c]
        comps = ch.components[:k, :].astype(np.float64, copy=False)   # (k, WH)

        for s in range(S):
            X = pred_res[s, c].reshape(T, W * H).astype(np.float64, copy=False)  # (T, WH)
            if ch.center:
                X = X - ch.mean[None, :]

            scores = X @ comps.T   # (T, k)
            pred_weights[s, :, c, :k] = scores.astype(out_dtype, copy=False)
        x_rec = scores @ comps + ch.mean[None, :]

    return pred_weights, mode_mask


def project_weights_to_pixel_space(
    weights: np.ndarray,  # (S, T, C, Kmax) - ensemble only
    pca_model: PCAModel,
    n_modes_per_channel: Dict[int, int],
) -> np.ndarray:
    """
    Reconstruct pixel-space residual from calibrated ensemble PCA weights.

    Parameters
    ----------
    weights : np.ndarray
        (S, T, C, Kmax) - calibrated predicted residual PCA weights
    pca_model : PCAModel
    n_modes_per_channel : dict[channel] = k

    Returns
    -------
    res_recon : np.ndarray
        (S, C, T, W, H)
    """
    if weights.ndim == 3:
        # (T, C, Kmax) -> (1, T, C, Kmax)
        weights = np.expand_dims(weights, axis=0)
    S, T, C, Kmax = weights.shape
    W, H = pca_model.per_channel[0].spatial_shape
    res_recon = np.zeros((S, C, T, W, H), dtype=np.float32)

    for c in range(C):
        ch    = pca_model.per_channel[c]
        k     = n_modes_per_channel[c]
        comps = ch.components[:k, :].astype(np.float64)  # (k, W*H)
        mean  = ch.mean.astype(np.float64)                # (W*H,)

        for s in range(S):
            w   = weights[s, :, c, :k].astype(np.float64)  # (T, k)
            rec = w @ comps                                      # (T, W*H)
            if ch.center:
                rec = rec + mean[None, :]
            res_recon[s, c] = rec.reshape(T, W, H)
    return res_recon

def main():
    parser = argparse.ArgumentParser(description="Fit adaptive residual calibrator")
    parser.add_argument(
        "--training_nc", type=str,
        default="/data/experiment_outputs/calibration_purpose/regression_2018-2019_all_samples_trained_on_2018-2019.nc",
    )
    parser.add_argument(
        "--validation_nc", type=str,
        default="/data/experiment_outputs/calibration_purpose/regression_2020_all_samples_trained_on_2018-2019.nc",
    )
    parser.add_argument("--path_pca_model", type=str, default="./pca/pca_all_samples_mode_100.npz")
    # parser.add_argument("--path_weight_pairs", type=str, default="./pca/val_pca_weight_pairs.npz")
    parser.add_argument("--path_weight_pairs", type=str, default="/data/experiment_outputs/calibration_purpose/pca_weights_2018-2019_and_2020.npz")
    parser.add_argument("--n_modes", type=int, default=100)
    parser.add_argument(
        "--max_time_idx", type=int, default=None,
        help="Maximum number of time steps to use for validation. Default None.",
    )
    parser.add_argument("--is_lr", type=bool, default=False)
    parser.add_argument("--mean_center", type=bool, default=True)
    args = parser.parse_args()

    save_dir = os.path.dirname(args.path_pca_model)
    os.makedirs(save_dir, exist_ok=True)
    # --- Do PCA on training data ---
    inp_train, gt_res_train, _ = load_training_data(args.training_nc, max_time_idx=args.max_time_idx)
    inp_val, gt_res_val, _ = load_training_data(args.validation_nc, max_time_idx=args.max_time_idx)
    gt_res = np.concatenate([gt_res_train, gt_res_val], axis=1)
    inp = np.concatenate([inp_train, inp_val], axis=1)
    # for low resolution conditions 
    if args.is_lr:
        gt_res = inp
        gt_res_train = inp_train
        gt_res_val = inp_val
    
    pca_model = fit_truncated_pca_per_channel(
        gt_res,
        n_components=args.n_modes,
        center=args.mean_center,
        n_iter=5,
        random_state=0,
        save_dtype=np.float32,
    )
    save_pca_model(pca_model, args.path_pca_model)
    model = load_pca_model(args.path_pca_model)
    summary = summarize_pca_model(model)

    for c, info in summary.items():
        print(f"Channel {c}:")
        print(f"  effective_rank_topk = {info['effective_rank_topk']:.3f}")
        print(f"  captured_variance_ratio_topk = {info['captured_variance_ratio_topk']:.6f}")
        print(f"  n_components_stored = {int(info['n_components_stored'])}")
    
    # --- Fit on validation ---

    # res, gt_res, gt, reg, pred, inp = load_validation_data(args.validation_nc, max_time_idx=args.max_time_idx, return_all=True)
    dataset = build_pca_weight_pair_dataset2(
        gt_res_train=gt_res_train,
        gt_res_val=gt_res_val,
        model=model,
        use_effective_rank=False,
        round_effective_rank="ceil",
        # n_modes_per_channel={0: 19, 1: 23, 2: 21, 3: 25, 4: 19, 5: 23, 6: 19, 7: 22, 8: 21, 9: 21},
        out_dtype=np.float32,
    )

    gt_weights_source = dataset["gt_weights_source"]         # (T, C, Kmax)
    gt_weights_target = dataset["gt_weights_target"]         # (T, C, Kmax)
    mode_mask = dataset["mode_mask"]           # (C, Kmax)
    n_modes_per_channel = dataset["n_modes_per_channel"]
    
    save_weight_pair_dataset2(
        args.path_weight_pairs,
        gt_weights_source=gt_weights_source,
        gt_weights_target=gt_weights_target,
        mode_mask=mode_mask,
        n_modes_per_channel=n_modes_per_channel,
    )
if __name__ == "__main__":
    main()