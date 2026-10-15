"""
Data preparation: caches spherical_particles.csv, registration_shifts.csv,
PGT samples, the noise model, and noise-injected PGT variants.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import tifffile
import scipy.ndimage
import scipy.optimize
import concurrent.futures
from tqdm import tqdm

#
# add src/ to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ablation_config import (
    DATA_CACHE_DIR, PGT_DIR, NOISE_COEFFS_JSON,
    SPHERICAL_PARTICLES_CSV, REGISTRATION_SHIFTS_CSV,
    DETECTION_SPECIMEN,
)
from _1_constants import TIFF_MAX_INTENSITY_VALUE
from _2_data_loading_helpers import (
    load_intra_sample_registration_data, parse_data_structure,
)

# __ 1. Spherical particles CSV ___________________________________________

def ensure_spherical_particles_csv(data_root: str, force: bool = False) -> str:
    """Create spherical_particles.csv via Cellpose if it doesn't exist."""
    if os.path.exists(SPHERICAL_PARTICLES_CSV) and not force:
        print(f"[data_prep] spherical_particles.csv already cached at {SPHERICAL_PARTICLES_CSV}")
        return SPHERICAL_PARTICLES_CSV

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    print("[data_prep] Running spherical particle detection (Cellpose)…")

    # import here so the module can be imported without cellpose installed
    from tools.detect_spherical_particles import (
        preprocess_image, extract_particles_from_mask,
        filter_circles_with_intersecting_overlap,
    )
    from cellpose import models as cellpose_models
    import cv2

    triplets = parse_data_structure(data_root)
    triplets = [t for t in triplets if DETECTION_SPECIMEN in t['specimen_dir']]
    if not triplets:
        raise ValueError(f"No triplets found for specimen '{DETECTION_SPECIMEN}'")

    cellpose_model = cellpose_models.CellposeModel(gpu=True)

    all_particles = []
    for triplet in tqdm(triplets, desc="Detecting spherical particles"):
        triplet_id = triplet["ID"]
        img = preprocess_image(cv2.imread(triplet["T2"], cv2.IMREAD_ANYDEPTH))

        masks_large, _, _ = cellpose_model.eval(
            cv2.GaussianBlur(img, (13, 13), 0), diameter=220, resample=True,
            niter=500, flow_threshold=1.0, cellprob_threshold=0.0)
        masks_small, _, _ = cellpose_model.eval(
            img, diameter=15, resample=True,
            niter=500, flow_threshold=0.3, cellprob_threshold=0.0)

        raw = (
            extract_particles_from_mask(masks_large, group_id=0, min_circularity=0.85) +
            extract_particles_from_mask(masks_small, group_id=1, min_circularity=0.85)
        )
        detector_particles = [p for p in raw if not (p["group_id"] == 0 and p["diameter_px"] < 150)]
        final = filter_circles_with_intersecting_overlap(detector_particles)

        for draw_order, p in enumerate(sorted(final, key=lambda p: p["diameter_px"], reverse=True)):
            all_particles.append({
                "triplet_ID": triplet_id, "cx_px": p["cx_px"], "cy_px": p["cy_px"],
                "diameter_px": p["diameter_px"], "draw_order": draw_order,
            })

    import pandas as pd
    df = pd.DataFrame(all_particles)
    df.to_csv(SPHERICAL_PARTICLES_CSV, index=False)
    print(f"[data_prep] Saved {len(df)} particles to {SPHERICAL_PARTICLES_CSV}")
    return SPHERICAL_PARTICLES_CSV


# __ 2. Registration shifts CSV ____________________________________________

def ensure_registration_csv(data_root: str, force: bool = False) -> str:
    """Create registration_shifts.csv via phase cross-correlation if it doesn't exist."""
    if os.path.exists(REGISTRATION_SHIFTS_CSV) and not force:
        print(f"[data_prep] registration_shifts.csv already cached at {REGISTRATION_SHIFTS_CSV}")
        return REGISTRATION_SHIFTS_CSV

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    print("[data_prep] Running intra-sample registration…")

    from datetime import datetime
    from skimage.registration import phase_cross_correlation
    from tools.utils import estimate_snr_acldr

    all_triplets = parse_data_structure(data_root)
    for t in all_triplets:
        time_string = t['ID'][:19]
        t['datetime'] = datetime.strptime(time_string, "%Y-%m-%d_%H-%M-%S")
    all_triplets.sort(key=lambda x: (x['specimen_dir'], x['datetime']))

    grouped = {}
    sample_counter = -1
    current_dir = None
    dir_item_count = 0
    for t in all_triplets:
        if t['specimen_dir'] != current_dir:
            current_dir = t['specimen_dir']
            sample_counter += 1
            dir_item_count = 0
        if dir_item_count > 0 and dir_item_count % 4 == 0:
            sample_counter += 1
        t['sample_group_ID'] = sample_counter
        grouped.setdefault(sample_counter, []).append(t)
        dir_item_count += 1

    all_registrations = []
    for group_triplets in tqdm(grouped.values(), desc="Processing sample groups"):
        processed = []
        for t in group_triplets:
            img_etd = tifffile.imread(t['ETD'])
            img_t1 = tifffile.imread(t['T1'])
            img_t2 = tifffile.imread(t['T2'])
            avg = (img_etd.astype(np.float64) + img_t1.astype(np.float64) + img_t2.astype(np.float64)) / 3.0
            t['snr_of_avg_img'] = estimate_snr_acldr(avg, 4)
            t['image_data'] = avg
            processed.append(t)
        if not processed:
            continue
        processed.sort(key=lambda x: -x['snr_of_avg_img'])
        ref = processed[0]['image_data']
        for t in processed:
            shift, _, _ = phase_cross_correlation(ref, t['image_data'], upsample_factor=1000)
            t['reg_shift_Y'] = shift[0]
            t['reg_shift_X'] = shift[1]
            all_registrations.append({
                "triplet_ID": t["ID"], "sample_group_ID": t["sample_group_ID"],
                "reg_shift_X": t["reg_shift_X"], "reg_shift_Y": t["reg_shift_Y"],
                "snr_of_avg_img": t["snr_of_avg_img"],
            })
            del t['image_data']

    import pandas as pd
    df = pd.DataFrame(all_registrations)
    df.to_csv(REGISTRATION_SHIFTS_CSV, index=False)
    print(f"[data_prep] Saved {len(df)} records to {REGISTRATION_SHIFTS_CSV}")
    return REGISTRATION_SHIFTS_CSV


# __ 3. Pseudo-ground-truth synthesis ______________________________________

DETECTOR_NAMES = ['ETD', 'T1', 'T2']


def _update_crop_bounds(shift, bounds, shape):
    ct, cb, cl, cr = bounds
    H, W = shape
    dy, dx = shift
    ct = max(ct, int(np.ceil(max(0, dy))))
    cb = min(cb, H - int(np.ceil(max(0, -dy))))
    cl = max(cl, int(np.ceil(max(0, dx))))
    cr = min(cr, W - int(np.ceil(max(0, -dx))))
    return ct, cb, cl, cr


def ensure_pgt(data_root: str, reg_csv: str, force: bool = False) -> str:
    """Synthesise pseudo-ground-truth samples from registered data."""
    # check if PGT already exists (at least one sample dir with clean/ subfolder)
    existing = glob.glob(os.path.join(PGT_DIR, "*", "clean"))
    if existing and not force:
        print(f"[data_prep] PGT already cached at {PGT_DIR} ({len(existing)} samples)")
        return PGT_DIR

    os.makedirs(PGT_DIR, exist_ok=True)
    print("[data_prep] Synthesising pseudo-ground-truth…")

    from skimage.filters import threshold_otsu

    reg_data = load_intra_sample_registration_data(reg_csv)
    all_triplets = parse_data_structure(data_root, None, reg_data)

    # group by sample_group_ID, skip detection specimen
    samples = {}
    for t in all_triplets:
        if DETECTION_SPECIMEN in t['specimen_dir']:
            continue
        sid = t.get('sample_group_ID')
        if sid is not None:
            samples.setdefault(sid, []).append(t)

    def _preprocess(sample_id, group):
        group_snrs = []
        shifts = []
        for t in group:
            ri = reg_data[t['ID']]
            group_snrs.append(ri['snr_of_avg_img'])
            shifts.append((ri['shift_Y'], ri['shift_X']))
        shape = tifffile.imread(group[0][DETECTOR_NAMES[0]]).shape
        H, W = shape
        crop = (0, H, 0, W)
        for s in shifts:
            crop = _update_crop_bounds(s, crop, (H, W))
        # snr-weighted (ACLDR) average, mathematically identical to
        # sum(w_i * x_i) / sum(w_i). dividing all weights by their maximum first
        # is overflow-free (the ACLDR estimator can degenerate to float64-max SNR
        # and summing such weights overflows) and does not change the result.
        w = np.asarray(group_snrs, dtype=np.float64)
        finite_max = np.max(w[np.isfinite(w)]) if np.any(np.isfinite(w)) else 0.0
        if finite_max > 0:
            # non-finite weights take the largest finite value (dominant image,
            # matching the formula's limit)
            w = np.where(np.isfinite(w), w, finite_max)
            w = w / w.max()
            w = w / w.sum()
        else:
            w = np.zeros(len(group), dtype=np.float64)

        results = {}
        for det in DETECTOR_NAMES:
            num = np.zeros(shape, dtype=np.float64)
            for idx, t in enumerate(group):
                img = tifffile.imread(t[det])
                shifted = scipy.ndimage.shift(img, shift=shifts[idx], mode='constant', cval=0.0)
                num += shifted * w[idx]
            results[det] = num
        return sample_id, {
            'specimen_name': os.path.basename(group[0]['specimen_dir']),
            'max_sample_snr': np.max(group_snrs),
            'crop_bounds': crop,
            'pgt_imgs': results,
        }

    pipeline_data = {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = {ex.submit(_preprocess, sid, grp): sid for sid, grp in samples.items()}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preprocessing samples"):
            sid, data = f.result()
            pipeline_data[sid] = data

    # otsu threshold on max snr; outliers excluded from the otsu input (this also
    # keeps degenerate snr estimates from breaking the histogram) but remain
    # eligible for pgt saving via the threshold check.
    raw_snrs = np.array([d['max_sample_snr'] for d in pipeline_data.values()])
    q1, q3 = np.percentile(raw_snrs, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = {k: v for k, v in pipeline_data.items() if lo <= v['max_sample_snr'] <= hi}
    filtered_snrs = np.array([d['max_sample_snr'] for d in filtered.values()])
    otsu_thresh = threshold_otsu(filtered_snrs[filtered_snrs > 0])

    valid = [sid for sid, d in pipeline_data.items() if d['max_sample_snr'] >= otsu_thresh]
    print(f"[data_prep] Otsu threshold={otsu_thresh:.2f}, {len(valid)} valid samples")

    def _save(sample_id, data):
        base = os.path.join(PGT_DIR, f"{data['specimen_name']}_sample{sample_id:02d}")
        clean = os.path.join(base, "clean")
        os.makedirs(clean, exist_ok=True)
        ct, cb, cl, cr = data['crop_bounds']
        for det in DETECTOR_NAMES:
            tifffile.imwrite(os.path.join(clean, f"{det}.tiff"),
                             data['pgt_imgs'][det][ct:cb, cl:cr].astype(np.uint16))

    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = [ex.submit(_save, sid, pipeline_data[sid]) for sid in valid]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Saving PGTs"):
            f.result()

    print(f"[data_prep] PGT saved to {PGT_DIR}")
    return PGT_DIR


# __ 4. Noise scaling model ________________________________________________

def ensure_noise_model(data_root: str, reg_csv: str, pgt_dir: str, force: bool = False) -> str:
    """Compute per-detector noise scaling polynomial coefficients."""
    if os.path.exists(NOISE_COEFFS_JSON) and not force:
        print(f"[data_prep] noise_model_coefficients.json already cached")
        return NOISE_COEFFS_JSON

    print("[data_prep] Computing detector noise scaling…")
    reg_data = load_intra_sample_registration_data(reg_csv)
    all_triplets = parse_data_structure(data_root, None, reg_data)

    samples = {}
    for t in all_triplets:
        sid = t.get('sample_group_ID')
        if sid is not None:
            samples.setdefault(sid, []).append(t)

    num_bins = 500
    bins = np.linspace(0, TIFF_MAX_INTENSITY_VALUE, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    sum_sq_res = {det: np.zeros(num_bins, dtype=np.float64) for det in DETECTOR_NAMES}
    counts = {det: np.zeros(num_bins, dtype=np.float64) for det in DETECTOR_NAMES}

    for sample_id, group in tqdm(samples.items(), desc="Analyzing noise scaling"):
        specimen_name = os.path.basename(group[0]['specimen_dir'])
        clean_dir = os.path.join(pgt_dir, f"{specimen_name}_sample{sample_id:02d}", "clean")
        if not os.path.isdir(clean_dir):
            continue

        shifts = []
        for t in group:
            ri = reg_data[t['ID']]
            shifts.append((ri['shift_Y'], ri['shift_X']))

        shape = tifffile.imread(group[0][DETECTOR_NAMES[0]]).shape
        H, W = shape
        crop = (0, H, 0, W)
        for s in shifts:
            crop = _update_crop_bounds(s, crop, (H, W))
        ct, cb, cl, cr = crop

        for det in DETECTOR_NAMES:
            pgt_path = os.path.join(clean_dir, f"{det}.tiff")
            if not os.path.exists(pgt_path):
                continue
            pgt_clean = tifffile.imread(pgt_path).astype(np.float64)
            bin_idx = np.clip(np.digitize(pgt_clean, bins) - 1, 0, num_bins - 1)
            for idx, t in enumerate(group):
                raw = tifffile.imread(t[det]).astype(np.float64)
                shifted = scipy.ndimage.shift(raw, shift=shifts[idx], mode='constant', cval=0.0)
                cropped = shifted[ct:cb, cl:cr]
                residual = cropped - pgt_clean
                np.add.at(sum_sq_res[det], bin_idx, residual ** 2)
                np.add.at(counts[det], bin_idx, 1)

    model_coeffs = {}
    for det in DETECTOR_NAMES:
        valid = counts[det] > 0
        if np.any(valid):
            x = bin_centers[valid] / TIFF_MAX_INTENSITY_VALUE
            y = (sum_sq_res[det][valid] / counts[det][valid]) / (TIFF_MAX_INTENSITY_VALUE ** 2)
            p0 = np.polyfit(x, y, 3)
            res = scipy.optimize.least_squares(
                lambda p, x, y: np.polyval(p, x) - y, p0, args=(x, y), loss='soft_l1')
            model_coeffs[det] = res.x.tolist()

    with open(NOISE_COEFFS_JSON, "w") as f:
        json.dump(model_coeffs, f, indent=4)
    print(f"[data_prep] Noise model saved to {NOISE_COEFFS_JSON}")
    return NOISE_COEFFS_JSON


# __ 5. Noise injection ___________________________________________________

def ensure_noise_injection(pgt_dir: str, noise_coeffs_json: str, force: bool = False) -> None:
    """Inject noise into PGT samples at multiple sigma levels and realizations."""
    # check if any noisy dirs exist
    existing_noisy = glob.glob(os.path.join(pgt_dir, "*", "noisy_sigma*"))
    if existing_noisy and not force:
        print(f"[data_prep] Noise-injected samples already cached ({len(existing_noisy)} dirs)")
        return

    print("[data_prep] Injecting noise into PGT samples…")

    noise_steps = 30
    noise_realizations = 15
    noise_sigma_max = 2.0
    sigmas = (np.linspace(0, np.sqrt(noise_sigma_max), noise_steps) ** 2)[1:]

    noise_coeffs = None
    global_max_base_sigma = 1.0
    if os.path.exists(noise_coeffs_json):
        with open(noise_coeffs_json, "r") as f:
            noise_coeffs = json.load(f)
        x_eval = np.linspace(0, 1.0, 1000)
        max_sigmas = [np.max(np.sqrt(np.maximum(np.polyval(p, x_eval), 0.0))) for p in noise_coeffs.values()]
        if max_sigmas:
            global_max_base_sigma = max(max_sigmas)

    def _inject(sample_dir):
        clean_dir = os.path.join(sample_dir, "clean")
        if not os.path.isdir(clean_dir):
            return
        clean_imgs = {}
        for det in DETECTOR_NAMES:
            p = os.path.join(clean_dir, f"{det}.tiff")
            if os.path.exists(p):
                clean_imgs[det] = tifffile.imread(p).astype(np.float64)
            else:
                return
        for sigma in sigmas:
            for rel in range(noise_realizations):
                noisy_dir = os.path.join(sample_dir, f"noisy_sigma{sigma:.3f}_rel{rel:02d}")
                os.makedirs(noisy_dir, exist_ok=True)
                for det in DETECTOR_NAMES:
                    pgt = clean_imgs[det]
                    norm_I = pgt / TIFF_MAX_INTENSITY_VALUE
                    if noise_coeffs and det in noise_coeffs:
                        base_var = np.polyval(noise_coeffs[det], norm_I)
                        base_sigma = np.sqrt(np.maximum(base_var, 0.0))
                        local_sigma = sigma * (base_sigma / global_max_base_sigma)
                    else:
                        local_sigma = sigma * np.sqrt(norm_I)
                    noise = np.random.normal(0, local_sigma * TIFF_MAX_INTENSITY_VALUE, size=pgt.shape)
                    noisy = np.clip(np.round(pgt + noise), 0, TIFF_MAX_INTENSITY_VALUE).astype(np.uint16)
                    tifffile.imwrite(os.path.join(noisy_dir, f"{det}.tiff"), noisy)

    sample_dirs = [d for d in glob.glob(os.path.join(pgt_dir, "*")) if os.path.isdir(d)]
    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = [ex.submit(_inject, d) for d in sample_dirs]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Injecting noise"):
            f.result()
    print("[data_prep] Noise injection complete")


# __ Top-level entry points ________________________________________________

def ensure_training_data(data_root: str) -> tuple:
    """Ensure spherical_particles.csv and registration_shifts.csv exist."""
    particles_csv = ensure_spherical_particles_csv(data_root)
    reg_csv = ensure_registration_csv(data_root)
    return particles_csv, reg_csv


def ensure_evaluation_data(data_root: str, reg_csv: str) -> str:
    """Ensure PGT, noise model, and noise-injected samples exist. Returns PGT dir."""
    pgt_dir = ensure_pgt(data_root, reg_csv)
    noise_json = ensure_noise_model(data_root, reg_csv, pgt_dir)
    ensure_noise_injection(pgt_dir, noise_json)
    return pgt_dir


# __ CLI entry point ________________________________________________________

def main():
    parser = argparse.ArgumentParser(description="Prepare / cache all data artefacts for the ablation pipeline.")
    parser.add_argument('--data_root', type=str, required=True, help="Root of the experimental dataset (trinity_sem_dataset/)")
    parser.add_argument('--force', action='store_true', help="Re-create all artefacts even if cached.")
    parser.add_argument('--skip_pgt', action='store_true', help="Skip PGT / noise injection (training-only prep).")
    args = parser.parse_args()

    particles_csv, reg_csv = ensure_training_data(args.data_root)
    print(f"\n[done] particles_csv = {particles_csv}")
    print(f"[done] reg_csv      = {reg_csv}")

    if not args.skip_pgt:
        pgt_dir = ensure_evaluation_data(args.data_root, reg_csv)
        print(f"[done] pgt_dir      = {pgt_dir}")


if __name__ == "__main__":
    main()
    