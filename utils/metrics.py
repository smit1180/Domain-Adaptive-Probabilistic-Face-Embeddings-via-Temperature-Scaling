# ============================================================
# utils/metrics.py - Evaluation metrics
# 1:1 Verification (pair-based) + fusion utilities
# ============================================================
import numpy as np
from sklearn.metrics import roc_curve


def cosine_similarity(mu_i, mu_j):
    """Cosine similarity matrix. [N,D] x [M,D] -> [N,M]"""
    norm_i = mu_i / (np.linalg.norm(mu_i, axis=1, keepdims=True) + 1e-10)
    norm_j = mu_j / (np.linalg.norm(mu_j, axis=1, keepdims=True) + 1e-10)
    return (norm_i @ norm_j.T).astype(np.float32)


def mutual_likelihood_score(mu_i, sigma_i, mu_j, sigma_j):
    """
    MLS (Eq. 3 of PFE paper).
    sigma_i, sigma_j are standard deviations (not variance).
    Returns: [N, M] score matrix.
    """
    N, M = mu_i.shape[0], mu_j.shape[0]
    scores = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        diff_sq = (mu_i[i] - mu_j) ** 2
        var_sum = sigma_i[i]**2 + sigma_j**2 + 1e-10
        scores[i] = -0.5 * np.sum(diff_sq / var_sum + np.log(var_sum), axis=1)
    return scores.astype(np.float32)


def pairwise_mls_score(mu_a, sig_a, mu_b, sig_b):
    """
    Pairwise MLS for 1:1 verification.
    mu_a, mu_b: [N, D], sig_a, sig_b: [N, D]
    Returns: [N] scores (one per pair).
    """
    var_sum = sig_a**2 + sig_b**2 + 1e-10
    return -0.5 * np.sum(
        (mu_a - mu_b)**2 / var_sum + np.log(var_sum), axis=1)


def pair_mls_score(x1, x2):
    """Pairwise MLS for verification. x = concat(mu, sigma_sq)."""
    x1, x2 = np.array(x1), np.array(x2)
    D = x1.shape[1] // 2
    mu1, sig_sq1 = x1[:, :D], x1[:, D:]
    mu2, sig_sq2 = x2[:, :D], x2[:, D:]
    var_sum = sig_sq1 + sig_sq2 + 1e-10
    dist = np.sum(np.square(mu1 - mu2) / var_sum + np.log(var_sum), axis=1)
    return -dist


def pair_cosine_score(x1, x2):
    """Pairwise cosine for verification."""
    x1, x2 = np.array(x1), np.array(x2)
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist


def verification_accuracy(scores, labels, threshold=None):
    """Binary verification accuracy at given or optimal threshold."""
    if threshold is None:
        best_acc, best_th = 0, 0
        for th in np.sort(scores):
            preds = scores >= th
            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc, best_th = acc, th
        return best_acc, best_th
    else:
        preds = scores >= threshold
        return np.mean(preds == labels), threshold


def tar_at_far_from_pairs(scores, labels, far_target=0.01):
    """
    TAR@FAR for 1:1 verification from pre-computed pairs.
    scores: [N] similarity scores
    labels: [N] binary (1=genuine, 0=impostor)
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(far_target, fpr, tpr))


def compute_verification_metrics(scores, labels):
    """
    Compute all 1:1 verification metrics from pair scores.
    scores: [N] similarity scores
    labels: [N] binary (1=genuine, 0=impostor)
    Returns dict with: accuracy, TAR@FAR=1%, TAR@FAR=0.1%
    """
    acc, thresh = verification_accuracy(scores, labels)
    tar_1 = tar_at_far_from_pairs(scores, labels, 0.01)
    tar_01 = tar_at_far_from_pairs(scores, labels, 0.001)
    return {
        'accuracy': acc,
        'threshold': thresh,
        'tar@far=1%': tar_1,
        'tar@far=0.1%': tar_01,
    }


def fuse_pfe(mu_list, sigma_list):
    """
    Precision-weighted Bayesian fusion of PFE embeddings.
    Uses dimension-wise min-sigma + precision-weighted mean.
    """
    mu_arr    = np.stack(mu_list, axis=0)
    sigma_arr = np.stack(sigma_list, axis=0)
    sigma_fused = np.min(sigma_arr, axis=0)
    precision   = 1.0 / (sigma_arr**2 + 1e-10)
    total_prec  = precision.sum(axis=0)
    mu_fused    = (precision * mu_arr).sum(axis=0) / (total_prec + 1e-10)
    return mu_fused, sigma_fused
