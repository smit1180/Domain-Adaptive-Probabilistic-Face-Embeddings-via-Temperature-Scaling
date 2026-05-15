#!/usr/bin/env python
"""
step4_eval_proposed.py - Evaluate Strategy A (1:1 Verification)
Per-camera, all-combined, and fused results.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, '.')
import config
from step3_train_tau import DomainAdaptationHead
from utils.metrics import (pairwise_mls_score,
                           compute_verification_metrics, fuse_pfe)


def load_model(strategy_id):
    path = os.path.join(config.MODELS_DIR,
                        'tau_strategy_{}.pt'.format(strategy_id))
    if not os.path.exists(path):
        print("  Model not found:", path)
        return None
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    scfg = ckpt['strategy']
    m = DomainAdaptationHead(
        ckpt.get('num_domains', config.NUM_DOMAINS),
        ckpt.get('embed_dim', config.EMBEDDING_DIM),
        multidim=scfg['multidim'],
        mu_correction=scfg['mu_correct']
    )
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m


def calibrate(model, mu, sigma, domain_ids):
    N = mu.shape[0]
    d = torch.zeros(N, config.NUM_DOMAINS)
    for i, did in enumerate(domain_ids):
        d[i, int(did)] = 1.0
    mu_t = torch.from_numpy(mu.astype(np.float32))
    s_t = torch.from_numpy(sigma.astype(np.float32))
    with torch.no_grad():
        mu_cal, sig_cal = model(mu_t, s_t, d)
    return mu_cal.numpy(), sig_cal.numpy()


def load_feat(path):
    d = np.load(path)
    cam = d['cam_ids'] if 'cam_ids' in d else \
        np.zeros(len(d['ids']), dtype=np.int32)
    return d['mu'], d['sigma'], d['ids'], cam


def filt(mu, sig, ids, cam, subjs):
    m = np.isin(ids, subjs)
    return mu[m], sig[m], ids[m], cam[m]


def fuse_raw(mu, sig, ids):
    """Fuse multiple images per subject."""
    uniq = np.unique(ids)
    mf, sf, idf = [], [], []
    for uid in uniq:
        m = ids == uid
        if m.sum() == 1:
            mf.append(mu[m][0])
            sf.append(sig[m][0])
        else:
            a, b = fuse_pfe(list(mu[m]), list(sig[m]))
            mf.append(a)
            sf.append(b)
        idf.append(uid)
    return (np.array(mf, np.float32),
            np.array(sf, np.float32),
            np.array(idf))


def build_pairs(ids_g, ids_p, n_impostor_ratio=1, seed=42):
    """Build genuine + impostor pairs for 1:1 verification."""
    rng = np.random.RandomState(seed)
    pairs = []
    for pi, pid in enumerate(ids_p):
        gi_list = np.where(ids_g == pid)[0]
        for gi in gi_list:
            pairs.append((gi, pi, 1))
    n_gen = len(pairs)
    for _ in range(n_gen * n_impostor_ratio):
        gi = rng.randint(len(ids_g))
        pi = rng.randint(len(ids_p))
        while ids_g[gi] == ids_p[pi]:
            pi = rng.randint(len(ids_p))
        pairs.append((gi, pi, 0))
    return pairs


def eval_strategy(strategy_id, model):
    """Evaluate a strategy with 1:1 verification."""
    scfg = config.STRATEGIES[strategy_id]
    print("\n  -- Strategy {}: {} --".format(
        strategy_id, scfg['name']))

    mu_g, sig_g, ids_g, cam_g = load_feat(
        os.path.join(config.FEATURES_DIR, 'mugshot_features.npz'))
    mu_p, sig_p, ids_p, cam_p = load_feat(
        os.path.join(config.FEATURES_DIR,
                     'surveillance_dist1_features.npz'))
    mu_g, sig_g, ids_g, cam_g = filt(
        mu_g, sig_g, ids_g, cam_g, config.TEST_SUBJ)
    mu_p, sig_p, ids_p, cam_p = filt(
        mu_p, sig_p, ids_p, cam_p, config.TEST_SUBJ)

    results = []

    # Calibrate all features
    mu_gc, sig_gc = calibrate(model, mu_g, sig_g, cam_g)
    mu_pc, sig_pc = calibrate(model, mu_p, sig_p, cam_p)

    # -- Per-camera 1:1 verification --
    for cam_id in sorted(np.unique(cam_p)):
        cm = cam_p == cam_id
        cam_name = (config.DOMAIN_NAMES[cam_id]
                    if cam_id < len(config.DOMAIN_NAMES)
                    else 'Cam{}'.format(cam_id))

        cam_pairs = build_pairs(ids_g, ids_p[cm])
        gi = np.array([p[0] for p in cam_pairs])
        pi = np.array([p[1] for p in cam_pairs])
        labels = np.array([p[2] for p in cam_pairs])

        scores = pairwise_mls_score(
            mu_gc[gi], sig_gc[gi],
            mu_pc[cm][pi], sig_pc[cm][pi])
        m = compute_verification_metrics(scores, labels)
        m.update({
            'method': scfg['name'],
            'strategy': strategy_id,
            'camera': cam_name,
            'cam_id': int(cam_id),
        })
        results.append(m)
        print("    {:<14} Acc={:.4f}  TAR@1%={:.4f}  "
              "TAR@0.1%={:.4f}".format(
                  cam_name, m['accuracy'],
                  m['tar@far=1%'], m['tar@far=0.1%']))

    # -- All cameras combined --
    all_pairs = build_pairs(ids_g, ids_p)
    gi = np.array([p[0] for p in all_pairs])
    pi = np.array([p[1] for p in all_pairs])
    labels = np.array([p[2] for p in all_pairs])

    scores = pairwise_mls_score(
        mu_gc[gi], sig_gc[gi], mu_pc[pi], sig_pc[pi])
    m = compute_verification_metrics(scores, labels)
    m.update({
        'method': scfg['name'],
        'strategy': strategy_id,
        'camera': 'All',
        'cam_id': -1,
    })
    results.append(m)
    print("    {:<14} Acc={:.4f}  TAR@1%={:.4f}  "
          "TAR@0.1%={:.4f}".format(
              'All', m['accuracy'],
              m['tar@far=1%'], m['tar@far=0.1%']))

    # -- Fused 1:1 verification --
    mu_pf, sig_pf, ids_pf = fuse_raw(mu_p, sig_p, ids_p)
    mu_gf, sig_gf, ids_gf = fuse_raw(mu_g, sig_g, ids_g)
    cam_p_dom = np.array([
        np.bincount(cam_p[ids_p == uid]).argmax()
        for uid in ids_pf], dtype=np.int32)
    cam_g_dom = np.zeros(len(ids_gf), dtype=np.int32)

    mu_gfc, sig_gfc = calibrate(model, mu_gf, sig_gf, cam_g_dom)
    mu_pfc, sig_pfc = calibrate(model, mu_pf, sig_pf, cam_p_dom)

    fused_pairs = build_pairs(ids_gf, ids_pf)
    gi = np.array([p[0] for p in fused_pairs])
    pi = np.array([p[1] for p in fused_pairs])
    labels = np.array([p[2] for p in fused_pairs])

    scores = pairwise_mls_score(
        mu_gfc[gi], sig_gfc[gi], mu_pfc[pi], sig_pfc[pi])
    m = compute_verification_metrics(scores, labels)
    m.update({
        'method': scfg['name'] + '+Fusion',
        'strategy': strategy_id,
        'camera': 'Fused',
        'cam_id': -2,
    })
    results.append(m)
    print("    {:<14} Acc={:.4f}  TAR@1%={:.4f}  "
          "TAR@0.1%={:.4f}".format(
              'Fused', m['accuracy'],
              m['tar@far=1%'], m['tar@far=0.1%']))

    return results


def main():
    print("=" * 65)
    print("STEP 4 -- Evaluate Strategy A (1:1 Verification)")
    print("=" * 65)

    all_results = []
    for sid in config.STRATEGIES:
        model = load_model(sid)
        if model is None:
            continue
        results = eval_strategy(sid, model)
        all_results.extend(results)

    path = os.path.join(config.RESULTS_DIR, 'proposed_results.npy')
    np.save(path, all_results)
    print("\n  Saved {} results to {}".format(
        len(all_results), path))
    print("\n" + "=" * 65)
    print("Step 4 complete")
    print("=" * 65)


if __name__ == '__main__':
    main()
