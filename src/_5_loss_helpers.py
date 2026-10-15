if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import torch

def dirichlet_energy(x):
    return ((x[:,:,1:]-x[:,:,:-1])**2).mean() + ((x[:,:,:,1:]-x[:,:,:,:-1])**2).mean()

def inter_channel_correlation(eta):
    b,c,h,w = eta.shape
    eta = eta.view(b,c,-1)
    # zero-mean
    eta = eta - eta.mean(2,True)
    cov = torch.bmm(eta,eta.transpose(1,2))/(h*w)
    off = cov*(1-torch.eye(c,device=eta.device)[None])
    return (off**2).mean()

def autocorrelation(eta, tau=1):
    diff_x = (eta[..., :, tau:] * eta[..., :, :-tau]).mean()**2
    diff_y = (eta[..., tau:, :] * eta[..., :-tau, :]).mean()**2
    return diff_x + diff_y

def sphere_normal_loss(nx, ny, nz, gt_normals, mask):
    mask_sum = mask.sum()
    if mask_sum < 1: return None
    # cosine similarity between predicted and GT normals
    dot = nx*gt_normals[:,0:1] + ny*gt_normals[:,1:2] + nz*gt_normals[:,2:3]
    return ((1.0 - dot) * mask).sum() / mask_sum

def uniform_composition_loss(c, mask):
    mask_sums = mask.sum(dim=(1, 2, 3), keepdim=True)
    valid = mask_sums.squeeze() > 0
    if valid.dim() == 0:
        valid = valid.unsqueeze(0)
    if not valid.any():
        return None

    # only batches with particle masks
    c_valid = c[valid]
    mask_valid = mask[valid]
    sum_valid = mask_sums[valid]
    # mean composition of the particles within each patch independently
    mean_c = (c_valid * mask_valid).sum(dim=(1, 2, 3), keepdim=True) / sum_valid

    # penalize variance from the patch's mean composition exclusively inside the mask
    variance = (((c_valid - mean_c)**2) * mask_valid).sum() / sum_valid.sum()
    return variance
