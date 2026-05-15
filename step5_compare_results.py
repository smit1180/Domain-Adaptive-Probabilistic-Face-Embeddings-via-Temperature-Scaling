#!/usr/bin/env python
"""
step5_compare_results.py - Generate 1:1 verification comparison table
Shows: MLS+fusion baseline, Strategy A (camera-wise + fusion)
Metrics: Accuracy, TAR@FAR=1%, TAR@FAR=0.1%
"""
import os, sys
import numpy as np

sys.path.insert(0, '.')
import config


def load():
    base = list(np.load(os.path.join(config.RESULTS_DIR,
        'baseline_results.npy'), allow_pickle=True))
    prop = list(np.load(os.path.join(config.RESULTS_DIR,
        'proposed_results.npy'), allow_pickle=True))
    return base, prop


def print_table(base, prop):
    """Print and save the 1:1 verification comparison table."""
    header = "{:<35} {:<14} {:>10} {:>10} {:>10}".format(
        'Method', 'Camera', 'Accuracy', 'TAR@1%', 'TAR@0.1%')
    sep = "=" * 82

    lines = []
    lines.append(header)
    lines.append(sep)

    # -- Baseline --
    lines.append("-- Baseline --")
    for r in base:
        lines.append("  {:<33} {:<14} {:10.4f} {:10.4f} {:10.4f}".format(
            r['method'], r.get('camera', 'All'),
            r['accuracy'], r['tar@far=1%'], r['tar@far=0.1%']))

    # -- Strategies A and D --
    for sid in ['A']:
        strat_results = [r for r in prop if r.get('strategy') == sid]
        if not strat_results:
            continue
        name = config.STRATEGIES[sid]['name']
        lines.append("")
        lines.append("-- Strategy {}: {} --".format(sid, name))
        for r in sorted(strat_results, key=lambda x: x.get('cam_id', 0)):
            cam = r.get('camera', 'All')
            lines.append("  {:<33} {:<14} {:10.4f} {:10.4f} {:10.4f}".format(
                r['method'], cam,
                r['accuracy'], r['tar@far=1%'], r['tar@far=0.1%']))

    lines.append(sep)

    # Print to console
    for line in lines:
        print(line)

    # Save to file
    path = os.path.join(config.RESULTS_DIR, 'comparison_table.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print("\n  Table saved to", path)


def main():
    print("=" * 65)
    print("STEP 5 -- 1:1 Verification Comparison Table")
    print("=" * 65)
    base, prop = load()
    print_table(base, prop)
    print("\n" + "=" * 65)
    print("Step 5 complete")
    print("=" * 65)


if __name__ == '__main__':
    main()
