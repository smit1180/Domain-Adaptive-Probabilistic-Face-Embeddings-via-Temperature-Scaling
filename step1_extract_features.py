#!/usr/bin/env python
"""
step1_extract_features.py - Extract mu, sigma from PFE checkpoint
Uses TensorFlow PFE model to extract 512-d embeddings and uncertainty.
"""
import os, sys, cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '.')
import config
from utils.dataset import (FaceDetector, get_mugshot_paths,
                           get_surveillance_by_distance,
                           get_surveillance_all)


class PFEModel:
    def __init__(self, ckpt_path):
        import tensorflow as tf
        tfv1 = tf.compat.v1
        tfv1.reset_default_graph()

        self.graph = tfv1.Graph()
        self.sess  = tfv1.Session(graph=self.graph)
        ckpt_dir   = os.path.dirname(ckpt_path)

        # Find meta file
        meta_file = None
        for c in [ckpt_path + '.meta', os.path.join(ckpt_dir, 'graph.meta')]:
            if os.path.exists(c):
                meta_file = c
                break
        if meta_file is None:
            raise FileNotFoundError(
                "No .meta file at {} or {}".format(
                    ckpt_path + '.meta',
                    os.path.join(ckpt_dir, 'graph.meta')))
        print("    Using meta file:", meta_file)

        with self.graph.as_default():
            # Load graph def directly (avoids saver/collection issues)
            from tensorflow.core.protobuf import meta_graph_pb2
            mgd = meta_graph_pb2.MetaGraphDef()
            with open(meta_file, 'rb') as f:
                mgd.ParseFromString(f.read())

            gd = mgd.graph_def
            # Remove training-only nodes
            skip_keywords = ['Adam', 'Momentum', 'save/', 'gradients/',
                             'train_op', 'beta1_power', 'beta2_power',
                             'group_deps', 'ExponentialMovingAverage']
            node_names = set()
            clean_nodes = []
            for n in gd.node:
                if any(s in n.name for s in skip_keywords):
                    continue
                clean_nodes.append(n)
                node_names.add(n.name)

            # Strip dangling control inputs
            for n in clean_nodes:
                bad = [inp for inp in n.input
                       if inp.startswith('^') and inp[1:] not in node_names]
                for b in bad:
                    n.input.remove(b)

            # Also clean regular inputs
            for n in clean_nodes:
                bad = [inp for inp in n.input
                       if not inp.startswith('^')
                       and inp.split(':')[0] not in node_names]
                # Don't remove regular inputs, just log
                # (these are usually placeholder/variable connections)

            # Build new graph_def
            from tensorflow.core.framework import graph_pb2
            new_gd = graph_pb2.GraphDef()
            new_gd.CopyFrom(gd)
            del new_gd.node[:]
            new_gd.node.extend(clean_nodes)

            print("    Graph: {} -> {} nodes".format(len(gd.node), len(clean_nodes)))

            # Import the cleaned graph def
            tf.import_graph_def(new_gd, name='')

            # Restore variables by directly assigning values from checkpoint
            reader = tfv1.train.NewCheckpointReader(ckpt_path)
            var_map = reader.get_variable_to_shape_map()

            # Find all tensors in graph that match checkpoint variables
            assign_ops = []
            restored = 0
            for vname, vshape in var_map.items():
                # Skip optimizer variables
                skip_kw = ['Adam', 'Momentum', 'beta1_power', 'beta2_power',
                           'ExponentialMovingAverage']
                if any(s in vname for s in skip_kw):
                    continue
                try:
                    tensor = self.graph.get_tensor_by_name(vname + ':0')
                    value = reader.get_tensor(vname)
                    assign_ops.append(tfv1.assign(tensor, value))
                    restored += 1
                except (KeyError, ValueError):
                    pass

            self.sess.run(assign_ops)
            print("    Restored {} variables from checkpoint".format(restored))

        # Find tensors
        names = [n.name for n in self.graph.as_graph_def().node]
        self.input_t = self._find(config.INPUT_TENSOR_NAMES, names)
        self.mu_t    = self._find(config.MU_TENSOR_NAMES, names)
        self.sigma_t = self._find(config.SIGMA_TENSOR_NAMES, names)
        self.phase_t = self._find(config.PHASE_TENSOR_NAMES, names, optional=True)

        print("    Input:", self.input_t.name if self.input_t is not None else "NONE")
        print("    Mu:   ", self.mu_t.name if self.mu_t is not None else "NONE")
        print("    Sigma:", self.sigma_t.name if self.sigma_t is not None else "NONE")
        print("    Phase:", self.phase_t.name if self.phase_t is not None else "NONE")

    def _find(self, candidates, names, optional=False):
        for c in candidates:
            base = c.replace(':0', '')
            if base in names:
                return self.graph.get_tensor_by_name(c)
        if optional:
            return None
        raise ValueError("Cannot find tensor from: {}".format(candidates))

    def extract(self, images):
        """images: [B, H, W, 3] float32 in [-1,1]"""
        fd = {self.input_t: images}
        if self.phase_t is not None:
            fd[self.phase_t] = False
        mu_raw, sig_raw = self.sess.run(
            [self.mu_t, self.sigma_t], feed_dict=fd)

        # Convert sigma tensor to std deviation
        n = self.sigma_t.name.lower()
        if 'log_sigma' in n or 'log_var' in n:
            sig_raw = np.exp(0.5 * np.clip(sig_raw, -10, 10))
        elif 'sigma_sq' in n or '_sq' in n:
            sig_raw = np.sqrt(np.abs(sig_raw) + 1e-10)
        sigma = np.abs(sig_raw) + 1e-8
        return mu_raw, sigma


def extract_features(model, detector, entries, desc=""):
    """Extract features for a list of (sid, cam_id, path) entries."""
    mus, sigs, ids, cams = [], [], [], []
    total = len(entries)
    for i, (sid, cam_id, path) in enumerate(entries):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print("  WARNING: cannot read {}".format(path))
            continue
        face = detector.crop(img_bgr, w=config.IMAGE_W, h=config.IMAGE_H)
        mu, sigma = model.extract(np.expand_dims(face, 0))
        mus.append(mu[0])
        sigs.append(sigma[0])
        ids.append(sid)
        cams.append(cam_id)
        if (i + 1) % 500 == 0:
            print("  {} {}/{}".format(desc, i + 1, total))

    return (np.array(mus, dtype=np.float32),
            np.array(sigs, dtype=np.float32),
            np.array(ids, dtype=np.int32),
            np.array(cams, dtype=np.int32))


def main():
    print("=" * 65)
    print("STEP 1 -- Extract Features")
    print("=" * 65)

    print("\n[Loading PFE model]")
    model = PFEModel(config.PFE_CHECKPOINT)

    det = FaceDetector()

    # --- Mugshots ---
    print("\n[1/2] Mugshot features ...")
    mugs = get_mugshot_paths(config.SCFACE_ROOT)
    if len(mugs) == 0:
        print("  ERROR: No mugshots found")
        sys.exit(1)

    mu_m, sig_m, ids_m, cam_m = extract_features(
        model, det, mugs, desc="  Mugshot")
    path = os.path.join(config.FEATURES_DIR, 'mugshot_features.npz')
    np.savez(path, mu=mu_m, sigma=sig_m, ids=ids_m, cam_ids=cam_m)
    print("  Saved {} shape={}".format(path, mu_m.shape))
    print("  sigma_mean={:.5f}".format(sig_m.mean()))

    # --- Surveillance (auto-detects cam_1 through cam_7) ---
    for i, dist in enumerate([1], start=2):
        print("\n[{}/2] Surveillance distance-{} ...".format(i, dist))

        surv = get_surveillance_by_distance(config.SCFACE_ROOT, dist)
        if len(surv) == 0:
            print("  Cam subfolders empty, trying surveillance_cameras_all ...")
            surv = get_surveillance_all(config.SCFACE_ROOT)
        if len(surv) == 0:
            print("  No images. Skipping.")
            continue

        print("  Total: {} images".format(len(surv)))
        from collections import Counter
        cam_counts = Counter(e[1] for e in surv)
        for c in sorted(cam_counts):
            print("    Cam{}: {} images".format(c, cam_counts[c]))

        mu_s, sig_s, ids_s, cam_s = extract_features(
            model, det, surv, desc="  Dist-{}".format(dist))
        path = os.path.join(config.FEATURES_DIR,
                            'surveillance_dist{}_features.npz'.format(dist))
        np.savez(path, mu=mu_s, sigma=sig_s, ids=ids_s, cam_ids=cam_s)
        print("  Saved {} shape={}".format(path, mu_s.shape))

        for c in sorted(set(cam_s)):
            m = cam_s == c
            print("    Cam{}: sigma_mean={:.5f} ({} imgs)".format(
                c, sig_s[m].mean(), m.sum()))

    print("\n" + "=" * 65)
    print("Step 1 complete")
    print("=" * 65)


if __name__ == '__main__':
    main()
