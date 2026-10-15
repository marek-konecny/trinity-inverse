if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import torch
import random
import tifffile
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import concurrent.futures
import torch.nn.functional as F
from torch.utils.data import Dataset
from _1_constants import PYTORCH_DEVICE, TIFF_MAX_INTENSITY_VALUE

class ScanTripletPatchSampler(Dataset):
    def __init__(self, triplets, patch_size, epoch_len):
        self.triplets = triplets
        self.patch_size = patch_size
        self.epoch_len = epoch_len
        self.pad = patch_size // 2
        self.data_cache = []

        with concurrent.futures.ThreadPoolExecutor() as ex:
            res = list(tqdm(ex.map(self._load, triplets), total=len(triplets), desc="Preloading triplets"))
        self.data_cache = [x for x in res if x is not None]

    def _precompute_normals(self, particles, h, w):
        normals = np.zeros((3,h,w), np.float32)
        normals[2] = 1.0
        mask = np.zeros((1,h,w), np.float32)

        if particles is None:
            return torch.from_numpy(normals), torch.from_numpy(mask)

        yy,xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        draw_order = particles[:, 3]
        draw_idx = np.argsort(draw_order, kind='stable')
        particles_ordered = particles[draw_idx]

        for p in particles_ordered:
            cx, cy, r = p[:3]
            dx = xx - cx
            dy = yy - cy
            d2 = dx*dx + dy*dy
            inside = d2 < r*r
            if not inside.any(): continue
            z = np.sqrt(r*r - d2[inside])
            normals[0][inside] = dx[inside] / r
            normals[1][inside] = dy[inside] / r
            normals[2][inside] = z / r
            mask[0][inside] = 1.0

        return torch.from_numpy(normals), torch.from_numpy(mask)

    def _load(self, triplet_info):
        img_etd = tifffile.imread(triplet_info['ETD']).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
        img_t1 = tifffile.imread(triplet_info['T1']).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
        img_t2 = tifffile.imread(triplet_info['T2']).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
        stack_np = np.stack([img_etd, img_t1, img_t2], 0)

        h,w = img_etd.shape
        n_torch, m_torch = self._precompute_normals(triplet_info['particles'], h, w)
        n_np = n_torch.numpy()
        m_np = m_torch.numpy()

        pad_amount = self.pad
        stack_pad = np.pad(stack_np, ((0,0), (pad_amount,pad_amount), (pad_amount,pad_amount)), mode='reflect')
        n_pad     = np.pad(n_np,     ((0,0), (pad_amount,pad_amount), (pad_amount,pad_amount)), mode='reflect')
        m_pad     = np.pad(m_np,     ((0,0), (pad_amount,pad_amount), (pad_amount,pad_amount)), mode='constant', constant_values=0)
        return torch.from_numpy(stack_pad), torch.from_numpy(n_pad), torch.from_numpy(m_pad)

    def __len__(self): return self.epoch_len

    def __getitem__(self, _):
        img, norm, mask = random.choice(self.data_cache)
        _, h_pad, w_pad = img.shape

        y = random.randint(0, max(0, h_pad - self.patch_size))
        x = random.randint(0, max(0, w_pad - self.patch_size))

        img_patch = img[:, y:y+self.patch_size, x:x+self.patch_size]

        return (
            img_patch,
            norm[:, y:y+self.patch_size, x:x+self.patch_size],
            mask[:, y:y+self.patch_size, x:x+self.patch_size]
        )

class ScanTripletWithFinetuningDataPatchSampler(Dataset):
    def __init__(self, triplets, reg_map, pretrained_model, patch_size, epoch_len):
        self.patch_size = patch_size
        self.epoch_len = epoch_len
        self.pad = patch_size // 2
        self.data_cache = []

        sample_groups = {}
        for triplet_info in triplets:
            group_id = triplet_info['sample_group_ID']
            if group_id not in sample_groups: sample_groups[group_id] = []
            sample_groups[group_id].append(triplet_info)

        pretrained_model.eval()

        for group_id, group in tqdm(sample_groups.items(), desc="Preloading & inferring finetune data"):
            if len(group) == 0: continue

            group_stacks = []
            stack_best = None
            
            best_triplet_id = max(group, key=lambda triplet_info: reg_map[triplet_info['ID']]['snr_of_avg_img'])['ID']

            for triplet_info in group:
                img_etd = tifffile.imread(triplet_info['ETD']).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
                img_t1 = tifffile.imread(triplet_info['T1']).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
                img_t2 = tifffile.imread(triplet_info['T2']).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
                stack = np.stack([img_etd, img_t1, img_t2], 0)

                shift_y = reg_map[triplet_info['ID']]['shift_Y']
                shift_x = reg_map[triplet_info['ID']]['shift_X']
                if shift_y != 0.0 or shift_x != 0.0:
                    for i in range(3):
                        stack[i] = scipy.ndimage.shift(stack[i], shift=[shift_y, shift_x], mode='constant', cval=0.0)

                group_stacks.append(stack)
                
                if triplet_info['ID'] == best_triplet_id: stack_best = stack

            with torch.inference_mode():
                t_in = torch.from_numpy(stack_best).unsqueeze(0).to(PYTORCH_DEVICE)
                pad_h = (8 - t_in.shape[2] % 8) % 8
                pad_w = (8 - t_in.shape[3] % 8) % 8
                if pad_h > 0 or pad_w > 0:
                    t_in = F.pad(t_in, (0, pad_w, 0, pad_h), mode='reflect')

                _, c, nx, ny, nz, _, _, _, _ = pretrained_model(t_in)

                if pad_h > 0 or pad_w > 0:
                    valid_h = c.shape[2] - pad_h
                    valid_w = c.shape[3] - pad_w
                    c = c[..., :valid_h, :valid_w]
                    nx = nx[..., :valid_h, :valid_w]
                    ny = ny[..., :valid_h, :valid_w]
                    nz = nz[..., :valid_h, :valid_w]

                c_np = c.squeeze(0).cpu().numpy()
                n_np = torch.cat([nx, ny, nz], dim=1).squeeze(0).cpu().numpy()

            #pad and cache all combinations
            pad_amount = self.pad
            c_pad = np.pad(c_np, ((0,0), (pad_amount,pad_amount), (pad_amount,pad_amount)), mode='reflect')
            n_pad = np.pad(n_np, ((0,0), (pad_amount,pad_amount), (pad_amount,pad_amount)), mode='reflect')
            for stack in group_stacks:
                stack_pad = np.pad(stack, ((0,0), (pad_amount,pad_amount), (pad_amount,pad_amount)), mode='reflect')
                self.data_cache.append((
                    torch.from_numpy(stack_pad),
                    torch.from_numpy(c_pad),
                    torch.from_numpy(n_pad)
                ))

    def __len__(self): return self.epoch_len

    def __getitem__(self, _):
        img, target_c, target_normals = random.choice(self.data_cache)
        _, h_pad, w_pad = img.shape
        y = random.randint(0, max(0, h_pad - self.patch_size))
        x = random.randint(0, max(0, w_pad - self.patch_size))

        return (
            img[:, y:y+self.patch_size, x:x+self.patch_size],
            target_c[:, y:y+self.patch_size, x:x+self.patch_size],
            target_normals[:, y:y+self.patch_size, x:x+self.patch_size]
        )
