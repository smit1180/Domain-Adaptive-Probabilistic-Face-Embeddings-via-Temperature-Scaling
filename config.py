# ============================================================
# config.py - Project Configuration
# PFE + tau-Calibration for Cross-Domain Face Recognition
# ============================================================
import os

# --- PATHS --------------------------------------------------------
# Original LFW aligned images (Person_Name/Person_Name_XXXX.jpg)
LFW_ALIGNED_DIR    = r"D:\BS Project\Dataset\lfw_aligned_96x112"

# LFW pairs file for standard verification protocol
LFW_PAIRS_PATH     = os.path.join(os.path.dirname(__file__), "proto", "lfw_pairs.txt")

# Generated synthetic CCTV dataset root
SCFACE_ROOT        = r"D:\BS Project\Dataset\lfw_C1_C2_C3_Dataset"

# PFE pretrained checkpoint
PFE_CHECKPOINT_DIR = r"D:\BS Project\Dataset\PFE_sphere64_casia_am"
PFE_CHECKPOINT     = os.path.join(PFE_CHECKPOINT_DIR, "ckpt-3000")

# --- OUTPUT DIRS --------------------------------------------------
FEATURES_DIR = "features"
RESULTS_DIR  = "results"
MODELS_DIR   = "saved_models"
for _d in [FEATURES_DIR, RESULTS_DIR, MODELS_DIR, "proto"]:
    os.makedirs(_d, exist_ok=True)

# --- PFE MODEL ----------------------------------------------------
IMAGE_H    = 112
IMAGE_W    = 96
IMAGE_SIZE = (IMAGE_W, IMAGE_H)   # (W, H) for cv2.resize
EMBEDDING_DIM = 512

# --- TENSOR NAMES -------------------------------------------------
INPUT_TENSOR_NAMES = ["images:0", "image_batch:0", "input:0"]
MU_TENSOR_NAMES    = ["SphereNet/l2_normalize_1:0",
                      "SphereNet/l2_normalize:0",
                      "embeddings:0", "l2_normalize:0"]
SIGMA_TENSOR_NAMES = ["sigma_sq:0", "sigma:0", "log_sigma_sq:0"]
PHASE_TENSOR_NAMES = ["phase_train:0", "is_training:0"]

# --- DATASET SPLIT ------------------------------------------------
NUM_SUBJECTS = 1680
TRAIN_SUBJ   = list(range(1, 1001))
TEST_SUBJ    = list(range(1001, 1681))

# --- DOMAINS (mugshot=0, cam1..3 = 1..3) --------------------------
NUM_DOMAINS  = 4
DOMAIN_NAMES = ['Mugshot', 'Cam1-VeryMild', 'Cam2-Mild', 'Cam3-MedMild']

# --- CAMERA DEGRADATION CONFIGS (3 cameras) -----------------------
CAMERA_CONFIGS = {
    1: {'name': 'VeryMild',   'blur_kernel': (3, 3), 'noise_std': 3},
    2: {'name': 'Mild',       'blur_kernel': (3, 3), 'noise_std': 8},
    3: {'name': 'MedMild',    'blur_kernel': (5, 5), 'noise_std': 12},
}

# --- TEMPERATURE HEAD ---------------------------------------------
TEMP_LR         = 0.01
TEMP_EPOCHS     = 300
TEMP_BATCH_SIZE = 128
DEVICE          = "cpu"

# --- TRAINING STRATEGY (A only) -----------------------------------
STRATEGIES = {
    'A': {'name': 'Scalar-tau',     'multidim': False, 'mu_correct': False, 'contrastive': False},
}
