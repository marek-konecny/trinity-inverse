if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import os
import json
import numpy as np
import scipy.signal
import scipy.linalg

def estimate_snr_acldr_return_variances(image: np.ndarray, autoregressive_order: int = 4) -> tuple[float, float, float]:
    """
    Approximates SNR of an image using the ACLDR method with a 2D Radial Autocorrelation Function (ACFR).

    [ Performance of signal-to-noise ratio estimation for scanning electron microscope using autocorrelation Levinson-Durbin recursion model ](https://doi.org/10.1111/jmi.12376)
    [ On-line identification of linear dynamic systems with applications to Kalman filtering ](https://doi.org/10.1109/TAC.1971.1099621)
    [ Parameter estimation of noisy autoregressive signals ](https://doi.org/10.1109/IRANIANCEE.2010.5507084)
    """
    I = image.astype(np.float64)
    H, W = I.shape
    mu_I = np.mean(I)
    
    I_centered = I - mu_I
    
    # 2D ACF using FFT (Wiener-Khinchin theorem)
    acf_2d_unnormalized = scipy.signal.fftconvolve(I_centered, I_centered[::-1, ::-1], mode='full')
    ones = np.ones_like(I)
    overlap_area = scipy.signal.fftconvolve(ones, ones[::-1, ::-1], mode='full')
    overlap_area[overlap_area == 0] = 1 
    acf_2d = acf_2d_unnormalized / overlap_area
    
    # radial distances from the center of ACF
    center_y, center_x = H - 1, W - 1
    y, x = np.indices(acf_2d.shape)
    distances = np.round(np.sqrt((y - center_y)**2 + (x - center_x)**2)).astype(int)
    
    # average ACFR values for each integer lag tau
    acfr = np.zeros(autoregressive_order + 1)
    for tau in range(autoregressive_order + 1):
        mask = (distances == tau)
        acfr[tau] = np.mean(acf_2d[mask])
            
    # solve via the Yule-Walker equations to find autoregressive coefficients (Levinson-Durbin equivalent)
    # R_mat * a_coeffs = -r_vector
    R_mat = scipy.linalg.toeplitz(acfr[0:autoregressive_order])
    r_vector = acfr[1:autoregressive_order+1]
    a_coeffs = scipy.linalg.solve(R_mat, -r_vector)
        
    # predict noise-free ACFR at lag 0
    acfr_0_hat = -np.sum(a_coeffs * acfr[1:autoregressive_order+1])
    
    # predict noise variance and signal variance
    sigma_n_sq = np.abs(acfr[0] - acfr_0_hat)
    sigma_s_sq = np.abs(acfr[1])
    
    # final SNR estimate
    snr_estimate = sigma_s_sq / sigma_n_sq
    return snr_estimate, sigma_s_sq, sigma_n_sq
def estimate_snr_acldr(image: np.ndarray, autoregressive_order: int = 4) -> float:
    snr_estimate, _, _ = estimate_snr_acldr_return_variances(image, autoregressive_order)
    return snr_estimate


def load_preinferred_latent_data(directory_path):
    """Expects 'params.json' and 'latents.npz' in the given directory."""
    with open(os.path.join(directory_path, "params.json"), "r") as parameters_file:
        det_parameters = json.load(parameters_file)

    det_p_vectors = np.array(det_parameters["det_p_vectors"], dtype=np.float32)
    alpha_logits = np.array(det_parameters["det_alpha_logits"], dtype=np.float32).flatten()
    det_alpha = 1.0 / (1.0 + np.exp(-alpha_logits))
    se_beta = np.array(det_parameters["se_beta"], dtype=np.float32).flatten()[0]
    det_se_w = np.array(det_parameters["det_se_w"], dtype=np.float32).flatten()
    det_se_b = np.array(det_parameters["det_se_b"], dtype=np.float32).flatten()

    loaded_data = np.load(os.path.join(directory_path, "latents.npz"))

    latent_c = loaded_data['comp'].astype(np.float32).squeeze()
    latent_normals = loaded_data['normals'].astype(np.float32)
    latent_r_lows = loaded_data['r_low'].astype(np.float32)
    latent_r_highs = loaded_data['r_high'].astype(np.float32)
    latent_gain = loaded_data['gain'].astype(np.float32).flatten()

    return (latent_c, latent_normals, latent_r_lows, latent_r_highs, det_alpha, det_se_w, det_se_b, se_beta, det_p_vectors, latent_gain)

def inference_from_latents(detector_index, rh_factor, rl_factor, det_p_rot_rad, latent_c, latent_normals, latent_r_lows, latent_r_highs, det_alpha, det_se_w, det_se_b, se_beta, det_p_vectors, latent_gain):
    px, py = det_p_vectors[detector_index]
    px_rotated = px * np.cos(det_p_rot_rad) - py * np.sin(det_p_rot_rad)
    py_rotated = px * np.sin(det_p_rot_rad) + py * np.cos(det_p_rot_rad)

    alpha_value = det_alpha[detector_index]
    w_se_value = det_se_w[detector_index]
    b_se_value = det_se_b[detector_index]
    detector_gain = latent_gain[detector_index]

    dot_product = latent_normals[0] * px_rotated + latent_normals[1] * py_rotated + latent_normals[2]
    bse_component = np.maximum(dot_product, 0)
    se_component = np.exp(se_beta * (1.0 - latent_normals[2]))

    shading_result = alpha_value * (w_se_value * latent_c + b_se_value) * se_component + (1.0 - alpha_value) * latent_c * bse_component

    if detector_index == 0: r_low_component = latent_r_lows[0]
    elif detector_index == 1: r_low_component = latent_r_lows[1]
    else: r_low_component = latent_r_lows[2]

    r_high_component = latent_r_highs[detector_index].squeeze()

    return (shading_result + (rl_factor * r_low_component)) * detector_gain + (rh_factor * r_high_component)
