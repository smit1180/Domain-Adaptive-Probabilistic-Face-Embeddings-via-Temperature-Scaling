# utils/dataset.py — Dataset loading utilities
import os
import numpy as np
import cv2
from typing import List, Tuple


class FaceDetector:
    def __init__(self):
        print("  Face detector: Images are pre-aligned. Passing through!")

    def crop(self, img_bgr, w=96, h=112):
        # Images are already 96x112 and aligned — just normalize
        face = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        return (face - 0.5) / 0.5


def get_mugshot_paths(scface_root: str) -> List[Tuple]:
    """Returns list of (subject_id, cam_id=0, img_path)."""
    mug_dir = os.path.join(scface_root, 'mugshot_frontal_cropped_all')
    if not os.path.isdir(mug_dir):
        raise FileNotFoundError(f"Not found: {mug_dir}")
    files = os.listdir(mug_dir)
    print(f"  Mugshot dir: {len(files)} files")
    EXTS = ['.JPG', '.jpg', '.jpeg', '.JPEG', '.png', '.PNG']
    entries = []
    for sid in range(1, 1700):
        for ext in EXTS:
            fp = os.path.join(mug_dir, f"{sid:03d}_frontal{ext}")
            if os.path.exists(fp):
                entries.append((sid, 0, fp))
                break
    print(f"  Found {len(entries)} mugshot images")
    return entries


def get_surveillance_by_distance(scface_root: str,
                                  distance: int) -> List[Tuple]:
    """
    Reads from surveillance_cameras_distance_{d}/cam_{c}/
    Returns list of (subject_id, cam_id, img_path)
    """
    dist_dir = os.path.join(scface_root,
                            f'surveillance_cameras_distance_{distance}')
    if not os.path.isdir(dist_dir):
        print(f"  Not found: {dist_dir}")
        return []

    EXTS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    entries    = []
    found_cams = []

    for cam_id in range(1, 8):
        cam_dir = os.path.join(dist_dir, f'cam_{cam_id}')
        if not os.path.isdir(cam_dir):
            continue
        found_cams.append(cam_id)

        for sid in range(1, 1700):
            found = False
            for idx in [1, 2, 3]:
                for ext in EXTS:
                    fp = os.path.join(cam_dir,
                                      f"{sid:03d}_cam{cam_id}_{idx}{ext}")
                    if os.path.exists(fp):
                        entries.append((sid, cam_id, fp))
                        found = True
                        break
                if found:
                    break
            if not found:
                for ext in EXTS:
                    fp = os.path.join(cam_dir,
                                      f"{sid:03d}_cam{cam_id}{ext}")
                    if os.path.exists(fp):
                        entries.append((sid, cam_id, fp))
                        break

    print(f"  Distance-{distance}: {len(entries)} images "
          f"from cams {found_cams}")
    return entries


def get_surveillance_all(scface_root: str) -> List[Tuple]:
    """Fallback: all images from surveillance_cameras_all."""
    surv_dir = os.path.join(scface_root, 'surveillance_cameras_all')
    if not os.path.isdir(surv_dir):
        return []
    EXTS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    entries = []
    for sid in range(1, 1700):
        for cam in range(1, 8):
            for idx in range(1, 4):
                for ext in EXTS:
                    fp = os.path.join(surv_dir,
                                      f"{sid:03d}_cam{cam}_{idx}{ext}")
                    if os.path.exists(fp):
                        entries.append((sid, cam, fp))
                        break
    print(f"  surveillance_cameras_all: {len(entries)} images")
    return entries


def preprocess_image(img_path: str, image_size=None):
    import config as cfg
    size = image_size if image_size is not None else cfg.IMAGE_SIZE
    if isinstance(size, int):
        size = (size, size)
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return (img - 0.5) / 0.5
