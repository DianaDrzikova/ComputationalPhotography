#####
# Implementation of: Zhang, L., and Wan, Y. Color-to-gray conversion based on boundary points. 2022
# Diana Maxima Drzikova, xdrzik01, 2023/2024
#####


import cv2
import numpy as np
from skimage import filters
from scipy.ndimage import sobel
from skimage.filters import threshold_local
import argparse


# compute color image gradients
def getGradients(img):

    L, A, B = cv2.split(img)

    matrix_x = np.array([[-1, 0, +1],
                    [-2, 0, +2],
                    [-1, 0, +1]])

    matrix_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [+1, +2, +1]])

    # Compute the mean of each component
    mean_stack = np.mean(img, axis=(0, 1))
    v_stack = np.var(img, axis=(0, 1))

    wc0 = mean_stack / (mean_stack[0]+mean_stack[1]+mean_stack[2])
    wc = v_stack / (v_stack[0]+v_stack[1]+v_stack[2])

    Ic = np.empty((3, img.shape[0], img.shape[1]), dtype=np.float32)

    for i, channel in enumerate([L, A, B]):
        Gx = cv2.filter2D(channel, -1, matrix_x)
        Gy = cv2.filter2D(channel, -1, matrix_y)
        Ic[i] = np.sqrt(Gx**2 + Gy**2)

    I_grad = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i, channel in enumerate(Ic):
        I_grad += cv2.filter2D(channel, -1, wc0+wc)

    return I_grad

# compute gradient for grayscale image
def getGradientsGrayscale(image_gray):
    Gx = sobel(image_gray, axis=0)
    Gy = sobel(image_gray, axis=1)
    
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)

    if gradient_magnitude.ndim == 3 and gradient_magnitude.shape[2] == 1:
        gradient_magnitude = gradient_magnitude.reshape(gradient_magnitude.shape[:2])
    
    if gradient_magnitude.max() > 1.0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
    
    threshold = filters.threshold_otsu(gradient_magnitude)
    
    boundary_points = gradient_magnitude > threshold
    
    return boundary_points

# apply threshold
def getBoundary(gradient_magnitude):
    threshold = np.percentile(gradient_magnitude, 82) # 82 is selected by experimenting
    boundary_points = gradient_magnitude > threshold
    return boundary_points

# get gray images
def getGray(img):
    b, g, r = cv2.split(img)

    i = 0.0
    j = 0.0

    gray = []

    while(i <= 1.0):
        wr = i
        j = 0.0
        while(j <= (1.0 - wr)):
            wg = j
            gray.append(wr * r + wg * g + (1 - wr - wg) * b)
            j+=0.1
        i+=0.1

    return np.array(gray)


# search for best match between boundary points
def find_optimal_grayscale(lab_boundary, gray_boundary):
    max_overlap = 0
    index = 0

    for i in range(len(gray_boundary)):
        candidate_grayscale = gray_boundary[i]

        overlap = np.sum(lab_boundary & candidate_grayscale)
        
        if overlap > max_overlap:
            index = i
            max_overlap = overlap
    
    return index

# final adjustion to match the needed range
def stretch_grayscale_values(grayscale_image):
    Min = np.min(grayscale_image)
    Max = np.max(grayscale_image)
    
    stretched = (grayscale_image - Min) / (Max - Min) * 255
    stretched_uint8 = np.uint8(stretched)
    
    return stretched_uint8


def convert(filename, out_filename):
    img = cv2.imread(filename)

    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img)

    # change range of A,B components based on the paper
    A_normalized = ((A + 128) * (100 / 255)).astype(np.uint8)
    B_normalized = ((B + 128) * (100 / 255)).astype(np.uint8)

    lab_image = cv2.merge([L, A_normalized, B_normalized])

    gray_boundary = []
    lab_gradients = getGradients(lab_image)
    lab_boundary = getBoundary(lab_gradients)
    gray = getGray(img)

    for g in gray:
        gray_gradients = getGradientsGrayscale(g)
        gray_boundary.append(gray_gradients)

    gray_boundary = np.array(gray_boundary)

    index = find_optimal_grayscale(lab_boundary, gray_boundary)
    output = stretch_grayscale_values(gray[index])

    print(f"Index of selected best gray image: {index}.")
    
    cv2.imwrite("boundary_selectedgray.png", gray_boundary[index].astype(np.int8)*255)
    cv2.imwrite("boundary_lab.png", lab_boundary.astype(np.int8)*255)
    cv2.imwrite(out_filename, output)


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

    convert(input, output)