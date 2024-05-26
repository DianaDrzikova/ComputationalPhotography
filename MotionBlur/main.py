import cv2
import numpy as np
from sklearn.cluster import KMeans
import scipy.sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from scipy import ndimage
import argparse

def compute_reblur(blurred_edge_image, sigma_0):
    return cv2.GaussianBlur(blurred_edge_image, (0, 0), sigma_0)

def compute_gradient(image):
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gradient_x**2 + gradient_y**2)

def compute_ratio(blurred_gradient, reblurred_gradient):
    return np.max(blurred_gradient) / np.max(reblurred_gradient)

def estimate_blur_amount(ratio, sigma_0):
    return 1 / np.sqrt(ratio**2 - 1) * sigma_0

def apply_bilateral_filter(sparse_blur_measurement, sigma_spatial, sigma_range):
    non_zero_mask = sparse_blur_measurement > 0
    filtered_result = cv2.bilateralFilter(sparse_blur_measurement, 9, sigma_range, sigma_spatial)
    
    filtered_result = filtered_result * non_zero_mask
    return filtered_result

def deblur(filename, out_filename):
    # read the image
    image = cv2.imread(filename)
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # blur amount estimation
    sigma_0 = 0.8

    # compute the blurred edge
    blurred_edge_image = compute_reblur(image, sigma_0)

    # compute gradients
    blurred_gradient = compute_gradient(image)

    reblurred_gradient = compute_gradient(blurred_edge_image)

    # compute ratio R
    ratio = compute_ratio(blurred_gradient, reblurred_gradient)
    # estimate sigma

    sigma = estimate_blur_amount(ratio, sigma_0)

    # blur amount refinement

    filtered = apply_bilateral_filter(blurred_edge_image, 1, sigma)

    filtered_gradient = compute_gradient(filtered)

    threshold = 20
    edge_mask = filtered_gradient > threshold

    pixels = filtered.reshape((-1, 1)) 
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)

    segmented_image = kmeans.labels_.reshape(blurred_edge_image.shape)

    for i in range(k):
        segmented_image[segmented_image == i] = kmeans.cluster_centers_[i]

    segmented_image = np.uint8(segmented_image)

    # blur amount propagation

    L = ndimage.laplace(image)

    laplacian_flat = L.flatten()

    edge_mask_flat = edge_mask.flatten().astype(np.float32)

    d_bar = filtered_gradient.flatten()

    D = scipy.sparse.diags(edge_mask_flat)
    L = scipy.sparse.diags(laplacian_flat)

    lambda_reg = 0.1

    A = L + lambda_reg * D.dot(D)
    b = lambda_reg * D.dot(d_bar)

    d = spsolve(A, b)

    full_defocus_map = d.reshape(image.shape)

    cv2.imwrite(out_filename, filtered_gradient)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', 
        type=str, 
        help='Filepath to color image')
    
    parser.add_argument(
        '--output', 
        type=str, 
        help='Filepath for gray image')
    
    args = parser.parse_args()
    input = args.input
    output = args.output

    deblur(input, output)