if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import cv2
import numpy as np

def solve_poisson_equation(vector_field_x, vector_field_y, target_boundary_image=None):
    divergence_field = (vector_field_x - np.roll(vector_field_x, 1, axis=1)) + (vector_field_y - np.roll(vector_field_y, 1, axis=0))
    target_image_float32 = np.zeros_like(divergence_field, dtype=np.float32) if target_boundary_image is None else target_boundary_image.astype(np.float32)
    regularization_lambda = 2e-5 
    forcing_function = divergence_field - regularization_lambda * target_image_float32
    padded_forcing_function = np.pad(forcing_function, ((0, forcing_function.shape[0]), (0, forcing_function.shape[1])), mode='symmetric')
    padded_height, padded_width = padded_forcing_function.shape
    frequency_u, frequency_v = np.meshgrid(np.fft.fftfreq(padded_width), np.fft.fftfreq(padded_height))
    frequency_denominator = (2 * np.cos(2 * np.pi * frequency_u) - 2) + (2 * np.cos(2 * np.pi * frequency_v) - 2) - regularization_lambda
    frequency_domain_solution = np.fft.fft2(padded_forcing_function) / frequency_denominator
    reconstructed_image = np.real(np.fft.ifft2(frequency_domain_solution))[:forcing_function.shape[0], :forcing_function.shape[1]]
    return reconstructed_image + (np.mean(target_image_float32) - np.mean(reconstructed_image))

def compute_itti_koch_saliency_map(source_image):
    internal_calculation_width, internal_calculation_height = 640, 480
    statistics_width, statistics_height = 40, 30
    original_height, original_width = source_image.shape[:2]

    resized_intensity_image = cv2.resize(source_image, (internal_calculation_width, internal_calculation_height), interpolation=cv2.INTER_AREA)

    def build_gaussian_pyramid(base_image, pyramid_depth=9):
        image_pyramid = [base_image]
        current_level_image = base_image
        for _ in range(pyramid_depth - 1):
            current_level_image = cv2.pyrDown(current_level_image)
            image_pyramid.append(current_level_image)
        return image_pyramid

    intensity_pyramid = build_gaussian_pyramid(resized_intensity_image)

    gabor_orientations_degrees = [0, 45, 90, 135]
    orientation_pyramids = {orientation_angle: [] for orientation_angle in gabor_orientations_degrees}
    gabor_kernel_size = 9
    gabor_sigma = 2.5
    gabor_lambda = 5.0
    gabor_gamma = 0.5
    gabor_psi = 0
    
    for orientation_angle in gabor_orientations_degrees:
        orientation_angle_radians = np.deg2rad(orientation_angle)
        gabor_filter_kernel = cv2.getGaborKernel((gabor_kernel_size, gabor_kernel_size), gabor_sigma, orientation_angle_radians, gabor_lambda, gabor_gamma, gabor_psi, ktype=cv2.CV_32F)
        for pyramid_level_image in intensity_pyramid:
            filtered_image = cv2.filter2D(pyramid_level_image, cv2.CV_32F, gabor_filter_kernel)
            orientation_pyramids[orientation_angle].append(np.abs(filtered_image))

    center_surround_scale_pairs = [(2,5), (2,6), (3,6), (3,7), (4,7), (4,8)]
    intensity_feature_maps = []
    orientation_feature_maps = []

    def compute_center_surround_difference(center_scale_map, surround_scale_map):
        map_height, map_width = center_scale_map.shape
        resized_surround_map = cv2.resize(surround_scale_map, (map_width, map_height), interpolation=cv2.INTER_LINEAR)
        return np.abs(center_scale_map - resized_surround_map)

    for center_scale_index, surround_scale_index in center_surround_scale_pairs:
        intensity_feature_maps.append(compute_center_surround_difference(intensity_pyramid[center_scale_index], intensity_pyramid[surround_scale_index]))
        for orientation_angle in gabor_orientations_degrees:
            orientation_feature_maps.append(compute_center_surround_difference(orientation_pyramids[orientation_angle][center_scale_index], orientation_pyramids[orientation_angle][surround_scale_index]))

    def apply_itti_normalization_operator(feature_map):
        normalized_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-12)
        downscaled_map_for_statistics = cv2.resize(normalized_map, (statistics_width, statistics_height), interpolation=cv2.INTER_AREA)
        dilated_map = cv2.dilate(downscaled_map_for_statistics, np.ones((3,3)))
        local_maxima_peaks = downscaled_map_for_statistics[(downscaled_map_for_statistics == dilated_map) & (downscaled_map_for_statistics > 0.0)]
        mean_of_local_maxima = local_maxima_peaks.mean() if len(local_maxima_peaks) > 0 else 0.0
        return normalized_map * ((downscaled_map_for_statistics.max() - mean_of_local_maxima)**2)

    def combine_feature_maps_to_conspicuity_map(feature_maps_list):
        combined_conspicuity_map = np.zeros((statistics_height, statistics_width), dtype=np.float32)
        for feature_map in feature_maps_list:
            resized_normalized_map = cv2.resize(apply_itti_normalization_operator(feature_map), (statistics_width, statistics_height), interpolation=cv2.INTER_LINEAR)
            combined_conspicuity_map += resized_normalized_map
        return combined_conspicuity_map
    
    final_saliency_map = (apply_itti_normalization_operator(combine_feature_maps_to_conspicuity_map(intensity_feature_maps)) + 
                            apply_itti_normalization_operator(combine_feature_maps_to_conspicuity_map(orientation_feature_maps))) / 2.0
    return cv2.resize(final_saliency_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
