#!/usr/bin/env python
"""
step2_eval_baseline.py - Baseline PFE 1:1 Verification
Generates genuine + impostor pairs, reports: Accuracy, TAR@FAR=1%, TAR@FAR=0.1%
"""
import os, sys
import numpy as np
from collections import Counter

sys.path.insert(0, '.')
import config
from utils.metrics import (pairwise_mls_score, compute_verification_metrics,
                           fuse_pfe)


def load(path):
    d = np.load(path)
    cam_ids = d['cam_ids'] if 'cam_ids' in d else np.zeros(len(d['ids']), dtype=np.int32)
    return d['mu'], d['sigma'], d['ids'], cam_ids


def filt(mu, sig, ids, cam, subjs):
    m = np.isin(ids, subjs)
    return mu[m], sig[m], ids[m], cam[m]


def build_pairs(ids_g, ids_p, n_impostor_ratio=1, seed=42):
    """
    Build genuine + impostor pair indices for 1:1 verification.
    Returns: list of (gallery_idx, probe_idx, label)
    """
    rng = np.random.RandomState(seed)
    pairs = []

    # Genuine pairs: same subject (mugshot <-> CCTV)
    for pi, pid in enumerate(ids_p):
        gi_list = np.where(ids_g == pid)[0]
        for gi in gi_list:
            pairs.append((gi, pi, 1))

    n_gen = len(pairs)

    # Impostor pairs: different subjects (random sampling)
    n_imp = n_gen * n_impostor_ratio
    for _ in range(n_imp):
        gi = rng.randint(len(ids_g))
        pi = rng.randint(len(ids_p))
        while ids_g[gi] == ids_p[pi]:
            pi = rng.randint(len(ids_p))
        pairs.append((gi, pi, 0))

    return pairs


def main():
    print("=" * 65)
    print("STEP 2 -- Baseline PFE 1:1 Verification")
    print("=" * 65)

    mu_g, sig_g, ids_g, cam_g = load(
        os.path.join(config.FEATURES_DIR, 'mugshot_features.npz'))
    mu_p, sig_p, ids_p, cam_p = load(
        os.path.join(config.FEATURES_DIR, 'surveillance_dist1_features.npz'))

    mu_g, sig_g, ids_g, cam_g = filt(mu_g, sig_g, ids_g, cam_g, config.TEST_SUBJ)
    mu_p, sig_p, ids_p, cam_p = filt(mu_p, sig_p, ids_p, cam_p, config.TEST_SUBJ)
    print("  Gallery: {} mugshots | Probe: {} surv".format(len(ids_g), len(ids_p)))

    cc = Counter(cam_p)
    print("  Probe cameras:", dict(sorted(cc.items())))

    # Build verification pairs
    pairs = build_pairs(ids_g, ids_p)
    gi_idx = np.array([p[0] for p in pairs])
    pi_idx = np.array([p[1] for p in pairs])
    labels = np.array([p[2] for p in pairs])
    n_gen = (labels == 1).sum()
    n_imp = (labels == 0).sum()
    print("  Pairs: {} genuine + {} impostor = {}".format(n_gen, n_imp, len(pairs)))

    results = []

    # --- MLS + fusion (vanilla) baseline ---
    print("\n  -- MLS + fusion (vanilla) --")
    scores = pairwise_mls_score(mu_g[gi_idx], sig_g[gi_idx],
                                mu_p[pi_idx], sig_p[pi_idx])
    m = compute_verification_metrics(scores, labels)
    m['method'] = 'PFE -- MLS + fusion (vanilla)'
    m['camera'] = 'All'
    results.append(m)
    print("    Accuracy: {:.4f}  TAR@1%: {:.4f}  TAR@0.1%: {:.4f}".format(
        m['accuracy'], m['tar@far=1%'], m['tar@far=0.1%']))

    # --- Per-camera breakdown ---
    print("\n  -- Per-camera breakdown --")
    for cam_id in sorted(np.unique(cam_p)):
        cam_name = config.DOMAIN_NAMES[cam_id] if cam_id < len(config.DOMAIN_NAMES) else 'Cam{}'.format(cam_id)
        # Build camera-specific pairs
        cam_mask_p = cam_p == cam_id
        cam_indices = np.where(cam_mask_p)[0]
        if len(cam_indices) == 0:
            continue
        cam_pairs = build_pairs(ids_g, ids_p[cam_mask_p])
        c_gi = np.array([p[0] for p in cam_pairs])
        c_pi = np.array([p[1] for p in cam_pairs])
        c_labels = np.array([p[2] for p in cam_pairs])

        mu_p_cam = mu_p[cam_mask_p]
        sig_p_cam = sig_p[cam_mask_p]
        c_scores = pairwise_mls_score(mu_g[c_gi], sig_g[c_gi],
                                      mu_p_cam[c_pi], sig_p_cam[c_pi])
        mc = compute_verification_metrics(c_scores, c_labels)
        print("    {:<14} Acc={:.4f}  TAR@1%={:.4f}  ({} probes)".format(
            cam_name, mc['accuracy'], mc['tar@far=1%'], cam_mask_p.sum()))

    path = os.path.join(config.RESULTS_DIR, 'baseline_results.npy')
    np.save(path, results)
    print("\n  Saved", path)
    print("\n" + "=" * 65)
    print("Step 2 complete")
    print("=" * 65)


if __name__ == '__main__':
    main()
