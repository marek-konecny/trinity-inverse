if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import sys
import argparse
import tifffile
import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import estimate_snr_acldr
from _2_data_loading_helpers import parse_data_structure


# 0. CONFIG
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_root', type=str, required=True)
parser.add_argument('--output_csv_path', type=str, default="intra_sample_registration.csv")
parser.add_argument('--acldr_autoregressive_order', type=int, default=4, help="Order P for the ACLDR method.")
parser.add_argument('--visualize_results', action='store_true', help="Show before/after registration overlays for the two most noisy images in first/last samples of each specimen.")
args = parser.parse_args()


# 1. HELPER METHODS
# ==========================================
def load_and_normalize_image(triplet_info):
    """Loads the 3 detector images, averages them, and normalizes to [0, 1] for RGB overlay."""
    img_etd = tifffile.imread(triplet_info['ETD']).astype(np.float64)
    img_t1 = tifffile.imread(triplet_info['T1']).astype(np.float64)
    img_t2 = tifffile.imread(triplet_info['T2']).astype(np.float64)
    avg_img = (img_etd + img_t1 + img_t2) / 3.0
    
    img_min = np.min(avg_img)
    img_max = np.max(avg_img)
    if img_max > img_min:
        avg_img = (avg_img - img_min) / (img_max - img_min)
    else:
        avg_img = np.zeros_like(avg_img)
    return avg_img

def create_colored_overlay(ref_img, mov_img):
    height, width = ref_img.shape
    overlay = np.zeros((height, width, 3), dtype=np.float32)
    overlay[..., 0] = ref_img  # RED channel
    overlay[..., 2] = ref_img  # BLU channel
    overlay[..., 1] = mov_img  # GRN channel
    return np.clip(overlay, 0, 1)


# 2. SETUP & SAMPLE GROUPING
# ==========================================
print("Parsing data structure and grouping triplets to samples...")
all_triplets = parse_data_structure(args.data_root)

# triplet datetime for chronological sorting
# TODO: in the future, similar images may be identified instead to make the script more generic
for triplet in all_triplets:
    time_string = triplet['ID'][:19]
    triplet['datetime'] = datetime.strptime(time_string, "%Y-%m-%d_%H-%M-%S")

all_triplets.sort(key=lambda x: (x['specimen_dir'], x['datetime']))

grouped_triplets = {}
sample_counter = -1
current_dir = None
dir_item_count = 0

for triplet in all_triplets:
    if triplet['specimen_dir'] != current_dir:
        current_dir = triplet['specimen_dir']
        sample_counter += 1
        dir_item_count = 0
        
    if dir_item_count > 0 and dir_item_count % 4 == 0:
        sample_counter += 1
        
    triplet['sample_group_ID'] = sample_counter
    
    if sample_counter not in grouped_triplets:
        grouped_triplets[sample_counter] = []
    grouped_triplets[sample_counter].append(triplet)
        
    dir_item_count += 1
print(f"Found {len(all_triplets)} triplets across {len(grouped_triplets)} sample groups.")


# 3. REGISTRATION & SNR ESTIMATION LOOP
# ==========================================
all_registrations = []

for group_id, group_triplets in tqdm(grouped_triplets.items(), desc="Processing sample groups"):
    processed_group = []
    
    for triplet in group_triplets:
        img_etd = tifffile.imread(triplet['ETD'])
        img_t1 = tifffile.imread(triplet['T1'])
        img_t2 = tifffile.imread(triplet['T2'])
        
        avg_img = (img_etd.astype(np.float64) + img_t1.astype(np.float64) + img_t2.astype(np.float64)) / 3.0
        
        estimated_snr = estimate_snr_acldr(avg_img, args.acldr_autoregressive_order)
        
        triplet['snr_of_avg_img'] = estimated_snr
        triplet['image_data'] = avg_img  
        
        processed_group.append(triplet)
        
    if not processed_group:
        continue
        
    # sort by SNR descending
    processed_group.sort(key=lambda x: -x['snr_of_avg_img'])
    
    # reference image implicitly receives an identity shift (0.0, 0.0)
    reference_image = processed_group[0]['image_data']
    
    for triplet in processed_group:
        shift, error, diffphase = phase_cross_correlation(reference_image, triplet['image_data'], upsample_factor=1000)
        
        triplet['reg_shift_Y'] = shift[0]
        triplet['reg_shift_X'] = shift[1]
        
        all_registrations.append({
            "triplet_ID": triplet["ID"],
            "sample_group_ID": triplet["sample_group_ID"],
            "reg_shift_X": triplet["reg_shift_X"],
            "reg_shift_Y": triplet["reg_shift_Y"],
            "snr_of_avg_img": triplet["snr_of_avg_img"]
        })
        del triplet['image_data']


# 4. EXPORT TO CSV
# ==========================================
df = pd.DataFrame(all_registrations)
df.to_csv(args.output_csv_path, index=False)
print(f"Saved {len(df)} records to {args.output_csv_path}")


# 5. VISUALIZATION
# ==========================================
if args.visualize_results:
    specimen_folders = {}
    for s_id, group in grouped_triplets.items():
        if not group: continue
        specimen_dir = group[0]['specimen_dir']
        if specimen_dir not in specimen_folders:
            specimen_folders[specimen_dir] = []
        specimen_folders[specimen_dir].append(s_id)

    for specimen_dir, group_ids in specimen_folders.items():
        group_ids = sorted(group_ids)
        
        picked_groups = []
        if len(group_ids) > 0: picked_groups.append(group_ids[0])
        if len(group_ids) > 1: picked_groups.append(group_ids[-1])
            
        for s_id in picked_groups:
            group = grouped_triplets[s_id]
            if len(group) < 2: continue
                
            sorted_by_noise = sorted(group, key=lambda x: x['snr_of_avg_img'])
            most_noisy_1 = sorted_by_noise[0]
            most_noisy_2 = sorted_by_noise[1]
            
            ref_img = load_and_normalize_image(most_noisy_2)
            mov_img = load_and_normalize_image(most_noisy_1)
                
            shift_y = most_noisy_1['reg_shift_Y'] - most_noisy_2['reg_shift_Y']
            shift_x = most_noisy_1['reg_shift_X'] - most_noisy_2['reg_shift_X']
            
            mov_img_shifted = scipy.ndimage.shift(mov_img, shift=[shift_y, shift_x], mode='constant', cval=0.0)
            
            overlay_before = create_colored_overlay(ref_img, mov_img)
            overlay_after = create_colored_overlay(ref_img, mov_img_shifted)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            short_dir = os.path.basename(specimen_dir)
            
            axes[0].imshow(overlay_before)
            axes[0].set_title(f"Before registration\nComputed shift: Y={shift_y:.3f}, X={shift_x:.3f}")
            axes[0].axis('off')
            
            axes[1].imshow(overlay_after)
            axes[1].set_title("After registration")
            axes[1].axis('off')
            
            plt.suptitle(f"Specimen: {short_dir} | Sample group: {s_id}\nMagenta: 2nd most noisy (SNR: {most_noisy_2['snr_of_avg_img']:.2f}) | Green: 1st most noisy (SNR: {most_noisy_1['snr_of_avg_img']:.2f})")
            plt.tight_layout()
            plt.show()
