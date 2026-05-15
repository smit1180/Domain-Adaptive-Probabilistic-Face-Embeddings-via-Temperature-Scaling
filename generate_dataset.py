#!/usr/bin/env python
# ============================================================
# generate_dataset.py — LFW → 3-Camera Synthetic CCTV Dataset
# Creates: 1 clean mugshot + 3 degraded CCTV per subject
# Camera noise levels: Very Mild, Mild, Medium Mild
# ============================================================
import os, cv2, glob
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────
LFW_DIR    = r"D:\BS Project\Dataset\lfw_aligned_96x112"
OUTPUT_DIR = r"D:\BS Project\Dataset\lfw_C1_C2_C3_Dataset"

CAMERA_CONFIGS = {
    1: {'name': 'VeryMild',   'blur_kernel': (3, 3), 'noise_std': 3},
    2: {'name': 'Mild',       'blur_kernel': (3, 3), 'noise_std': 8},
    3: {'name': 'MedMild',    'blur_kernel': (5, 5), 'noise_std': 12},
}


def apply_degradation(image, blur_kernel, noise_std):
    """Apply Gaussian blur + sensor noise to simulate CCTV."""
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    noise = np.random.normal(0, noise_std, blurred.shape)
    noisy = blurred + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():
    mugshot_dir = os.path.join(OUTPUT_DIR, "mugshot_frontal_cropped_all")
    os.makedirs(mugshot_dir, exist_ok=True)

    # Create cam directories
    cam_dirs = {}
    for cam_id, cfg in CAMERA_CONFIGS.items():
        d = os.path.join(OUTPUT_DIR, "surveillance_cameras_distance_1",
                         f"cam_{cam_id}")
        os.makedirs(d, exist_ok=True)
        cam_dirs[cam_id] = d

    # Find all subject folders with >= 2 images
    folders = sorted([f for f in glob.glob(os.path.join(LFW_DIR, "*"))
                      if os.path.isdir(f)])
    subject_id = 1

    print("Generating 3-camera synthetic CCTV dataset from LFW ...")
    print(f"  Source: {LFW_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Cameras: {len(CAMERA_CONFIGS)}")

    for folder in folders:
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        if len(images) < 2:
            continue

        # Image 1 → clean mugshot
        clean = cv2.imread(images[0])
        if clean is None:
            continue
        cv2.imwrite(os.path.join(mugshot_dir,
                    f"{subject_id:03d}_frontal.jpg"), clean)

        # Image 2 → degraded 3 ways (3 cameras)
        cctv_src = cv2.imread(images[1])
        if cctv_src is None:
            continue

        for cam_id, cfg in CAMERA_CONFIGS.items():
            degraded = apply_degradation(
                cctv_src, cfg['blur_kernel'], cfg['noise_std'])
            cv2.imwrite(
                os.path.join(cam_dirs[cam_id],
                             f"{subject_id:03d}_cam{cam_id}_1.jpg"),
                degraded)

        if subject_id % 200 == 0:
            print(f"  Processed {subject_id} subjects ...")
        subject_id += 1

    total = subject_id - 1
    print(f"\n{'='*50}")
    print(f"Dataset generation complete!")
    print(f"  Total subjects: {total}")
    print(f"  Mugshots: {total}")
    print(f"  CCTV images: {total * 3} ({total} × 3 cameras)")
    for cam_id, cfg in CAMERA_CONFIGS.items():
        print(f"    Cam{cam_id} ({cfg['name']}): "
              f"blur={cfg['blur_kernel']}, noise={cfg['noise_std']}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()