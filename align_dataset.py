import os
import cv2
import numpy as np
import math
import glob
from mtcnn import MTCNN

# We will read your synthetic dataset and save a perfectly aligned copy
INPUT_DIR = r"D:\BS Project\Dataset\archive (1)\lfw-deepfunneled\lfw-deepfunneled"
OUTPUT_DIR = r"D:\BS Project\Dataset\lfw_aligned_96x112"

def align_face(img, detector):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    
    if len(results) == 0:
        return None
        
    face = max(results, key=lambda b: b['box'][2] * b['box'][3])
    
    if 'keypoints' in face:
        left_eye = face['keypoints']['left_eye']
        right_eye = face['keypoints']['right_eye']
        
        # 1. Rotate to make eyes perfectly horizontal
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(dy, dx))
        
        # FIX: Force native Python integers for OpenCV
        cx = int((left_eye[0] + right_eye[0]) / 2)
        cy = int((left_eye[1] + right_eye[1]) / 2)
        eyes_center = (cx, cy)
        
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        img_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        
        # 2. Re-detect on the straightened face
        rot_rgb = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
        rot_results = detector.detect_faces(rot_rgb)
        if len(rot_results) > 0:
            face = max(rot_results, key=lambda b: b['box'][2] * b['box'][3])
            img = img_rotated

    # 3. Tight crop tailored for SphereFace (96x112)
    x, y, w, h = face['box']
    x, y = max(0, x), max(0, y)
    
    # Add margins to capture the full head
    mx, my = int(w * 0.15), int(h * 0.20)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return None
    
    # Final resize for PFE
    return cv2.resize(crop, (96, 112), interpolation=cv2.INTER_LINEAR)

def main():
    print("Initializing MTCNN (TF2)...")
    detector = MTCNN()
    
    # Replicate your folder structure
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if not file.endswith(('.jpg', '.png')): continue
            
            in_path = os.path.join(root, file)
            out_path = in_path.replace(INPUT_DIR, OUTPUT_DIR)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            img = cv2.imread(in_path)
            aligned = align_face(img, detector)
            
            # If MTCNN fails to find a face, just resize the original so we don't drop the subject
            if aligned is None:
                aligned = cv2.resize(img, (96, 112))
                
            cv2.imwrite(out_path, aligned)
            print(f"Aligned: {file}")

if __name__ == "__main__":
    main()