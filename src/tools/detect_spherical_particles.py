if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import sys
import cv2
import math
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from skimage import measure
# [ Cellpose-SAM: superhuman generalization for cellular segmentation ](https://doi.org/10.1101/2025.04.28.651001)
from cellpose import models as cellpose_models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _2_data_loading_helpers import parse_data_structure


# 0. CONFIG
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_root', type=str, required=True)
parser.add_argument('--detection_specimen_name', type=str, default="SiAl1")
parser.add_argument('--used_detector', type=str, default="T2", help="Which detector images to run particle detection on.")
parser.add_argument('--min_mask_circularity', type=float, default=0.85, help="Minimum circularity for Cellpose masks to be accepted as particles.")
parser.add_argument('--output_csv_path', type=str, default="spherical_particles.csv")
args = parser.parse_args()


# 1. SETUP
# ==========================================
triplets = parse_data_structure(args.input_data_root)
triplets = [t for t in triplets if args.detection_specimen_name in t['specimen_dir']]
if len(triplets) == 0: raise ValueError(f"No triplets found for specimen '{args.detection_specimen_name}'")

cellpose_model = cellpose_models.CellposeModel(gpu=True)


# 2. HELPER METHODS
# ==========================================
def preprocess_image(img):
    if img.dtype != np.uint16 and img.dtype != np.uint8: img = img.astype(np.uint16)

    img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # histogram normalization
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img8 = clahe.apply(img8)

    # cellpose expects float
    img = img8.astype(np.float32) / 255.0
    return img

def extract_particles_from_mask(mask, group_id, min_circularity):
    """
    Converts Cellpose mask to circles. Rejects zero-area or non-circular masks.
    """
    particles = []
    regions = measure.regionprops(mask)

    for r in regions:
        # circularity check
        if r.perimeter == 0:
            continue
        circularity = 4 * np.pi * r.area / (r.perimeter ** 2)
        if circularity < min_circularity:
            continue

        y0, x0 = r.centroid
        area = r.area

        entry = {
            "cx_px": x0,
            "cy_px": y0,
            "diameter_px": np.sqrt(4 * area / np.pi),
            "group_id": group_id
        }

        particles.append(entry)

    return particles

def filter_circles_with_intersecting_overlap(particles):
    """
    Iterates through particles in descending order:
    if a particle has intersecting overlap with any previously kept particle, discards it.
    Occluding overlap (outlines are, at most, internally tangent) is allowed.
    """
    
    particles = sorted(particles, key=lambda p: p["diameter_px"], reverse=True)
    kept = []

    for p in particles:
        xp, yp = p["cx_px"], p["cy_px"]
        rp = p["diameter_px"] / 2.0
        
        valid = True

        for q in kept:
            xq, yq = q["cx_px"], q["cy_px"]
            rq = q["diameter_px"] / 2.0

            # distance between centers
            dist = math.sqrt((xp - xq)**2 + (yp - yq)**2)

            # edges intersect: distance is strictly between the difference and sum of the radii
            if abs(rq - rp) < dist < (rp + rq):
                valid = False
                break

        if valid:
            kept.append(p)

    return kept


# 3. MAIN DETECTION LOOP
# ==========================================
all_particles = []
for triplet in tqdm(triplets, desc="Detecting spherical particles in scans"):
    triplet_id = triplet["ID"]
    
    img = preprocess_image(
        cv2.imread(triplet[args.used_detector], cv2.IMREAD_ANYDEPTH)
    )

    # cellpose pass 1 (large particles)
    masks_large, flows_large, styles_large = cellpose_model.eval(
        cv2.GaussianBlur(img,(13,13),0),
        diameter=220,
        resample=True,
        niter=500,
        flow_threshold=1.0,
        cellprob_threshold=0.0
    )

    # cellpose pass 2 (small particles)
    masks_small, flows_small, styles_small = cellpose_model.eval(
        img,
        diameter=15,
        resample=True,
        niter=500,
        flow_threshold=0.3,
        cellprob_threshold=0.0
    )
    
    raw_particles = (
        extract_particles_from_mask(masks_large, group_id=0, min_circularity=args.min_mask_circularity) + 
        extract_particles_from_mask(masks_small, group_id=1, min_circularity=args.min_mask_circularity)
    )

    # filter out small detections from the "large" pass
    detector_particles = [p for p in raw_particles if not (p["group_id"] == 0 and p["diameter_px"] < 150)] 

    final_particles = filter_circles_with_intersecting_overlap(detector_particles)

    for draw_order, p in enumerate(sorted(final_particles, key=lambda p: p["diameter_px"], reverse=True)):
        all_particles.append({
            "triplet_ID": triplet_id,
            "cx_px": p["cx_px"],
            "cy_px": p["cy_px"],
            "diameter_px": p["diameter_px"],
            "draw_order": draw_order
        })


# 4. EXPORT TO CSV
# ==========================================
df = pd.DataFrame(all_particles)
df.to_csv(args.output_csv_path, index=False)
print(f"saved {len(df)} particles to {args.output_csv_path}")
