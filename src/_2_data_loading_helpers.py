if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import os
import glob
import re
import numpy as np
import pandas as pd

def load_spherical_particles_data(csv_path):
    df = pd.read_csv(csv_path)
    particle_map = {}
    for triplet_id, group_df in df.groupby("triplet_ID"):
        group_df = group_df.sort_values('draw_order', kind='stable')
        particle_array = group_df[['cx_px', 'cy_px', 'diameter_px', 'draw_order']].values.astype(np.float32)
        particle_array[:, 2] *= 0.5
        particle_map[triplet_id] = particle_array
    return particle_map

def load_intra_sample_registration_data(csv_path):
    df = pd.read_csv(csv_path)
    reg_map = {}
    for _, row in df.iterrows():
        reg_map[row['triplet_ID']] = {
            'shift_Y': row['reg_shift_Y'], 'shift_X': row['reg_shift_X'],
            'sample_group_ID': int(row['sample_group_ID']),
            'snr_of_avg_img': float(row['snr_of_avg_img']) 
        }
    return reg_map

def parse_data_structure(root, spherical_particles_data=None, intra_sample_registration_data=None):
    # filename examples: 2024-01-15_14-30-0-ETD.tiff, 2024-01-15_14-30-1-TrinityT1.tiff, 2024-01-15_14-30-0-TrinityT2.tiff
    filename_regex = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-(\d+)-(.*)\.tiff")
    triplets = {}
    for f in glob.glob(os.path.join(root,"**","*.tiff"), recursive=True):
        match = filename_regex.match(os.path.basename(f))
        if not match: continue
        timestamp, sequence_num, detector = match.groups()
        triplet_id = f"{timestamp}-{sequence_num}"

        dname = os.path.dirname(f)
        if triplet_id not in triplets:
            triplets[triplet_id] = {'dir': dname}

        triplets[triplet_id][detector] = f

    parsed_triplets = []
    for triplet_id, detector_files in triplets.items():
        if all(x in detector_files for x in ['ETD', 'TrinityT1', 'TrinityT2']):
            particles = spherical_particles_data.get(triplet_id) if spherical_particles_data is not None else None
            
            sample_group_id = None
            if intra_sample_registration_data is not None and triplet_id in intra_sample_registration_data:
                sample_group_id = intra_sample_registration_data[triplet_id].get('sample_group_ID')

            parsed_triplets.append(dict(
                ETD=detector_files['ETD'], T1=detector_files['TrinityT1'], T2=detector_files['TrinityT2'],
                ID=triplet_id, particles=particles,
                specimen_dir=detector_files['dir'], sample_group_ID=sample_group_id
            ))

    return parsed_triplets
