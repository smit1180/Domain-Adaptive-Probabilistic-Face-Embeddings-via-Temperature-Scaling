# How to Run — PFE + τ-Calibration Pipeline

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `numpy`, `scipy`, `scikit-learn` — Core
- `opencv-python` — Image processing
- `tensorflow` (2.x with compat.v1) — PFE model inference
- `torch`, `torchvision` — τ-head training
- `matplotlib` — Training loss plot

### 2. Required Data

| Item | Path (set in `config.py`) |
|------|--------------------------|
| LFW aligned images (96×112) | `LFW_ALIGNED_DIR` |
| PFE checkpoint (`ckpt-3000`) | `PFE_CHECKPOINT_DIR` |
| Output dataset directory | `SCFACE_ROOT` |

### 3. Update Paths in `config.py`

Open `config.py` and update these three paths to match your machine:
```python
LFW_ALIGNED_DIR    = r"D:\BS Project\Dataset\lfw_aligned_96x112"
PFE_CHECKPOINT_DIR = r"D:\BS Project\Dataset\PFE_sphere64_casia_am"
SCFACE_ROOT        = r"D:\BS Project\Dataset\lfw_new_dataset_1"
```

---

## Step-by-Step Execution

Run each script **in order** from the project root directory:

### Step 0: Align Dataset
```bash
python align_dataset.py
```
- **Input**: Raw LFW images
- **Output**: Aligned face images (96×112) using MTCNN landmark detection
  - Rotation correction, tight cropping, and resizing
- **Time**: ~5–10 minutes

### Step 1: Generate Dataset
```bash
python generate_dataset.py
```
- **Input**: LFW aligned images
- **Output**: 3-camera synthetic CCTV dataset
  - `mugshot_frontal_cropped_all/` — Clean mugshots
  - `surveillance_cameras_distance_1/cam_1/` — Very Mild noise
  - `surveillance_cameras_distance_1/cam_2/` — Mild noise
  - `surveillance_cameras_distance_1/cam_3/` — Medium Mild noise
- **Time**: ~2–5 minutes

### Step 2: Extract Features
```bash
python step1_extract_features.py
```
- **Input**: Generated dataset + PFE checkpoint
- **Output**: `features/mugshot_features.npz`, `features/surveillance_dist1_features.npz`
- **Time**: ~10–30 minutes (depends on CPU/GPU)

### Step 3: Evaluate Baseline
```bash
python step2_eval_baseline.py
```
- **Input**: Extracted features
- **Output**: `results/baseline_results.npy` (MLS + fusion vanilla)
- **Time**: ~1 minute

### Step 4: Train τ-Head (Strategy A)
```bash
python step3_train_tau.py
```
- **Input**: Extracted features
- **Output**:
  - `saved_models/tau_strategy_A.pt` — Scalar-tau model
  - `results/tau_training_loss.png` — Training loss curve
- **Time**: ~5–10 minutes (CPU)

### Step 5: Evaluate Proposed Method
```bash
python step4_eval_proposed.py
```
- **Input**: Trained model + extracted features
- **Output**: `results/proposed_results.npy`
- **Time**: ~2 minutes

### Step 6: Generate Comparison Table
```bash
python step5_compare_results.py
```
- **Input**: Baseline + proposed results
- **Output**: `results/comparison_table.txt` — Final comparison table
- **Time**: < 1 minute

---

## Quick Run (All Steps)

```bash
python align_dataset.py
python generate_dataset.py
python step1_extract_features.py
python step2_eval_baseline.py
python step3_train_tau.py
python step4_eval_proposed.py
python step5_compare_results.py
```

---

## Expected Output

The final `results/comparison_table.txt` will contain:

```
Method                              Camera              Accuracy    T@1%    T@0.1%
==========================================================================================
-- Baseline --
  PFE -- MLS + fusion (vanilla)     All            0.XXXX   0.XXXX    0.XXXX

-- Strategy A: Scalar-tau --
  Scalar-tau+Fusion                 Fused           0.XXXX   0.XXXX    0.XXXX
  Scalar-tau                        All             0.XXXX   0.XXXX    0.XXXX
  Scalar-tau                        Cam1-VeryMild   0.XXXX   0.XXXX    0.XXXX
  Scalar-tau                        Cam2-Mild       0.XXXX   0.XXXX    0.XXXX
  Scalar-tau                        Cam3-MedMild    0.XXXX   0.XXXX    0.XXXX
==========================================================================================
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No .meta file` error in step1 | Check `PFE_CHECKPOINT_DIR` path in `config.py` |
| `No mugshots found` | Run `generate_dataset.py` first, check `SCFACE_ROOT` |
| `Model not found` in step4 | Run `step3_train_tau.py` first |
| TensorFlow import errors | Use `pip install tensorflow==2.12.0` |
| Low memory during feature extraction | Process in smaller batches (edit step1) |
