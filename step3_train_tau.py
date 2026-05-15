#!/usr/bin/env python
"""
step3_train_tau.py - Train tau-head (Strategy A: Scalar tau)
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
import config


# ==================================================================
# MODEL
# ==================================================================
class DomainAdaptationHead(nn.Module):
    """
    Unified tau-head for Strategy A.
    tau(domain) = exp(W_tau . one_hot + b_tau)   shape: [B, tau_dim]
    sigma_cal = sigma / tau
    """
    def __init__(self, num_domains=4, embed_dim=512,
                 multidim=False, mu_correction=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.multidim = multidim
        self.mu_correction = mu_correction
        tau_dim = embed_dim if multidim else 1
        self.W_tau = nn.Parameter(torch.zeros(num_domains, tau_dim))
        self.b_tau = nn.Parameter(torch.zeros(1, tau_dim))
        if mu_correction:
            self.W_mu = nn.Parameter(torch.zeros(num_domains, embed_dim))
            self.b_mu = nn.Parameter(torch.zeros(1, embed_dim))

    def get_tau(self, d):
        """d: [B, num_domains] one-hot -> tau: [B, tau_dim]"""
        return torch.exp(d @ self.W_tau + self.b_tau)

    def forward(self, mu, sigma, d):
        tau = self.get_tau(d)
        sigma_cal = sigma / tau
        if self.mu_correction:
            mu_cal = mu + d @ self.W_mu + self.b_mu
        else:
            mu_cal = mu
        return mu_cal, sigma_cal

    def get_tau_value(self, domain_id):
        self.eval()
        with torch.no_grad():
            d = torch.zeros(1, self.W_tau.shape[0])
            d[0, domain_id] = 1.0
            tau = self.get_tau(d)
            return tau.mean().item()

    def print_all_tau(self):
        print("  Learned tau per domain:")
        for i, name in enumerate(config.DOMAIN_NAMES):
            print("    {:<12}: tau_mean = {:.4f}".format(
                name, self.get_tau_value(i)))


# ==================================================================
# LOSS FUNCTIONS
# ==================================================================
def mls_loss(mu_a, sig_a, mu_b, sig_b):
    """MLS loss for genuine pairs. Minimize = maximize MLS."""
    var = sig_a**2 + sig_b**2 + 1e-10
    mls = -0.5 * (((mu_a - mu_b)**2 / var) + torch.log(var)).sum(1)
    return -mls.mean()


# ==================================================================
# DATASET
# ==================================================================
class PairDataset(Dataset):
    """Generates genuine pairs for training."""
    def __init__(self, mu_mug, sig_mug, ids_mug, cam_mug,
                 mu_sur, sig_sur, ids_sur, cam_sur,
                 train_subj):
        mm = np.isin(ids_mug, train_subj)
        ms = np.isin(ids_sur, train_subj)
        self.mu_m  = mu_mug[mm].astype(np.float32)
        self.sig_m = sig_mug[mm].astype(np.float32)
        self.ids_m = ids_mug[mm]
        self.cam_m = cam_mug[mm]
        self.mu_s  = mu_sur[ms].astype(np.float32)
        self.sig_s = sig_sur[ms].astype(np.float32)
        self.ids_s = ids_sur[ms]
        self.cam_s = cam_sur[ms]

        # Build genuine pairs
        self.pairs = []
        for i, sid in enumerate(self.ids_m):
            for j in np.where(self.ids_s == sid)[0]:
                self.pairs.append((i, j, 1))

        n_gen = len(self.pairs)
        print("  Pairs: {} genuine".format(n_gen))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mi, si, label = self.pairs[idx]
        nd = config.NUM_DOMAINS
        dm = torch.zeros(nd); dm[int(self.cam_m[mi])] = 1.0
        ds = torch.zeros(nd); ds[int(self.cam_s[si])] = 1.0
        return {
            'mu_a': torch.from_numpy(self.mu_m[mi]),
            'sig_a': torch.from_numpy(self.sig_m[mi]),
            'dom_a': dm,
            'mu_b': torch.from_numpy(self.mu_s[si]),
            'sig_b': torch.from_numpy(self.sig_s[si]),
            'dom_b': ds,
            'label': torch.tensor(label, dtype=torch.float32),
        }


# ==================================================================
# TRAINING
# ==================================================================
def train_strategy(strategy_id, dataset):
    """Train a single strategy and return model + history."""
    cfg = config.STRATEGIES[strategy_id]
    print("\n  -- Strategy {}: {} --".format(strategy_id, cfg['name']))

    device = torch.device(config.DEVICE)
    model = DomainAdaptationHead(
        config.NUM_DOMAINS, config.EMBEDDING_DIM,
        multidim=cfg['multidim'],
        mu_correction=cfg['mu_correct']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("  Parameters:", n_params)

    opt = optim.Adam(model.parameters(), lr=config.TEMP_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.TEMP_EPOCHS, eta_min=1e-6)
    loader = DataLoader(dataset, batch_size=config.TEMP_BATCH_SIZE,
                        shuffle=True, num_workers=0, drop_last=False)
    history = []

    for ep in range(1, config.TEMP_EPOCHS + 1):
        model.train()
        total = 0.0
        for b in loader:
            mu_a = b['mu_a'].to(device)
            sig_a = b['sig_a'].to(device)
            dom_a = b['dom_a'].to(device)
            mu_b = b['mu_b'].to(device)
            sig_b = b['sig_b'].to(device)
            dom_b = b['dom_b'].to(device)
            labels = b['label'].to(device)

            mu_a_cal, sig_a_cal = model(mu_a, sig_a, dom_a)
            mu_b_cal, sig_b_cal = model(mu_b, sig_b, dom_b)

            gm = labels == 1
            if gm.any():
                loss = mls_loss(mu_a_cal[gm], sig_a_cal[gm],
                               mu_b_cal[gm], sig_b_cal[gm])
            else:
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(mu_a)

        sched.step()
        avg = total / max(len(dataset), 1)
        history.append(avg)

        if ep % 50 == 0 or ep == 1:
            taus = "  ".join(
                "{}={:.3f}".format(config.DOMAIN_NAMES[i],
                                   model.get_tau_value(i))
                for i in range(config.NUM_DOMAINS))
            print("    Ep {:3d}  loss={:.4f}  {}".format(ep, avg, taus))

    return model, history


def load_features():
    """Load mugshot + surveillance features."""
    mug = np.load(os.path.join(config.FEATURES_DIR, 'mugshot_features.npz'))
    mu_m, sig_m = mug['mu'], mug['sigma']
    ids_m, cam_m = mug['ids'], mug['cam_ids']

    mu_all, sig_all, ids_all, cam_all = [], [], [], []
    for dist in [1]:
        path = os.path.join(config.FEATURES_DIR,
                            'surveillance_dist{}_features.npz'.format(dist))
        if not os.path.exists(path):
            continue
        d = np.load(path)
        mu_all.append(d['mu']); sig_all.append(d['sigma'])
        ids_all.append(d['ids']); cam_all.append(d['cam_ids'])

    mu_s = np.vstack(mu_all); sig_s = np.vstack(sig_all)
    ids_s = np.concatenate(ids_all); cam_s = np.concatenate(cam_all)

    print("\n  Mugshot: {}".format(mu_m.shape))
    print("  Surveillance: {}".format(mu_s.shape))
    print("  Cameras:", np.unique(cam_s))
    return mu_m, sig_m, ids_m, cam_m, mu_s, sig_s, ids_s, cam_s


def main():
    print("=" * 65)
    print("STEP 3 -- Train tau-Head (Strategy A)")
    print("=" * 65)

    mu_m, sig_m, ids_m, cam_m, mu_s, sig_s, ids_s, cam_s = load_features()

    # Build dataset (genuine pairs only — no contrastive needed)
    ds_gen = PairDataset(mu_m, sig_m, ids_m, cam_m,
                         mu_s, sig_s, ids_s, cam_s,
                         config.TRAIN_SUBJ)

    all_histories = {}

    for sid, scfg in config.STRATEGIES.items():
        print("\n" + "-" * 50)
        print("Training Strategy {}: {}".format(sid, scfg['name']))
        print("  mu_correct={}".format(scfg['mu_correct']))
        print("-" * 50)

        model, history = train_strategy(sid, ds_gen)
        all_histories[sid] = history

        # Save model
        path = os.path.join(config.MODELS_DIR,
                            'tau_strategy_{}.pt'.format(sid))
        torch.save({
            'model_state_dict': model.state_dict(),
            'strategy': scfg,
            'num_domains': config.NUM_DOMAINS,
            'embed_dim': config.EMBEDDING_DIM,
            'history': history,
        }, path)
        print("\n  Saved:", path)
        model.print_all_tau()

    # -- Plot training losses --
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'A': '#4C72B0'}
    for sid, hist in all_histories.items():
        ax.plot(hist, label="{}: {}".format(sid, config.STRATEGIES[sid]['name']),
                linewidth=2, color=colors.get(sid, '#888'))
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss -- Strategy A', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pp = os.path.join(config.RESULTS_DIR, 'tau_training_loss.png')
    plt.savefig(pp, dpi=150, bbox_inches='tight')
    print("\n  Plot saved:", pp)
    plt.close()

    print("\n" + "=" * 65)
    print("Step 3 complete -- Strategy A trained")
    print("=" * 65)


if __name__ == '__main__':
    main()
