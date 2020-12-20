import numpy as np
from scipy.ndimage.filters import correlate, median_filter
from imageio import imread
import matplotlib.pyplot as plt
import cv2

from .rectify_images import rectify_images

def get_disparity(It, Ib):
    """
    This function computes a stereo disparity image from top stereo 
    image It and bottom stereo image Ib. 

    Parameters:
    -----------
    It (np.ndarray): Top stereo image, m x n pixel np.array, greyscale.
    Ib (np.ndarray): Bottom stereo image, m x n pixel np.array, greyscale.

    Returns:
    --------
    Id (np.ndarray): Disparity image (map) as np.array, same size as It.
    """
    
    """ Algorithm used: Census Transform
    The paper I used is "Nonparametric Local Transforms for Computing Visual 
    Correspondence" (1994) by Ramin Zabih and John Woodfill.
    It can be found here: http://www.cs.cornell.edu/~rdz/Papers/ZW-ECCV94.pdf
    
    The algorithm operates identically to the naive SAD algorithm used in
    stereo_disparity_fast.py except that it operates not on the original images
    but instead on transformed images. These transformed images are transformed
    using the census transform. The census transform converts each pixel into a 
    bitstring representing whether the pixels in its immediate vicinity (a 
    transform patch) are less than the center pixel (1) or greater than the 
    center pixel (0). i.e. the census transform of 
    
        23 04 55
        01 32 99
        44 21 21
        
        would be 
        
        1 1 0
        1 0 0 
        0 1 1
        
        which is then flattened into a bitstring. 
    
    Then, rather than computing SAD error, the hamming distance between 
    bitstrings is used to compute error.
    
    This transform can tolerate a large number of outliers, improves detection
    around object boundaries, and ignores differences in brightness between the 
    two images. Most importantly, it is very fast. 
    """
 
    Il = It.T
    Ir = Ib.T

    Id = np.zeros(Il.shape)
    
    # Set all parameters
        # window: window size for SAD
        # transform_size: window size for census transform 
        # median_window_size: size of median filter used at the end
        # maxd: maximum disparity value; disparities must be within zero and maxd
    window = 20
    transform_size = 3
    median_window_size = 15
    maxd = 70
   
    height = Il.shape[0]
    width = Il.shape[1]

    # Getting useful pixels
    grad = np.gradient(Il)[1]
    min_gradient = 5

    kernel = np.ones((3,3),np.uint8)
    useful_pixels = cv2.dilate(np.uint8(np.abs(grad)>min_gradient), kernel, 1)

    # Census transform both left and right images
    Il_rt = census_transform(Il, transform_size).astype(int)
    Ir_rt = census_transform(Ir, transform_size).astype(int)
    
    all_pixel_errors = []
    all_patch_errors = []
    mask = np.ones((window, window))
    
    # Compute right image XOR'ed by the left image shifted by every possible disparity
    for disparity in range(-maxd, 1):
        all_pixel_errors.append(np.bitwise_xor(Ir_rt, np.roll(Il_rt, disparity, axis=1)) )
    
    # Compute the hamming distance for every XOR'ed set of images
    all_pixel_errors = np.vectorize(hamming_distance)(all_pixel_errors)

    # Aggregate pixel errors in a patch using correlate filter
    for i in range(len(all_pixel_errors)):
        all_patch_errors.append(correlate(all_pixel_errors[i], mask))
    
    # Loop through every pixel in image
    for y in range(height):
        for x in range(width):
            # Check if gradient is sufficiently large
            if useful_pixels[y][x] == 0:
                continue

            # Initialize minimum error and best disparity
            min_err = float('inf')
            best_disparity = x
            
            # For every possible disparity value
            for disparity in range(-maxd, 1):
                # Check if shifted pixel is within image
                if x+disparity < 0 or x+disparity >= width:
                    continue

                # Find the error for that disparity value and pixel
                err = all_patch_errors[disparity + maxd][y, x+disparity]
                
                # If the error value is less than the smallest one found so 
                #   far, set the newest lowest error and best disparity
                if err < min_err:
                    min_err = err
                    best_disparity = disparity
            
            # Set this pixel's disparity to the best disparity found
            Id[y][x] = -best_disparity
    
    # Use the median filter to smooth out disparity values
    Id = median_filter(Id, size = median_window_size)

    Id[Id >= maxd - 10] = 0

    return Id.T


def census_transform(I, transform_size):
    """
    This function applies the census transform to an image. 

    Parameters:
    -----------
    I (np.ndarray): an image, m x n pixel np.array, greyscale.
    transform_size (int): The window size of the transform

    Returns:
    --------
    I_ct (np.ndarray): a m x n np.array representing the census-transformed image
    """
    
    I_ct = np.zeros(I.shape)
    
    height = I.shape[0]
    width = I.shape[1]
    
    # Looping over every pixel in an image
    for y in range(height):
        for x in range(width):       
            # Find the bounds of the transform
            min_y = max(0, y-transform_size)
            max_y = min(y+transform_size, height-1)
            min_x = max(0, x-transform_size)
            max_x = min(x+transform_size, width-1)
            
            # Check if every pixel is greater than or less than the center pixel, 
            #   and flatten the resulting matrix of 1s and 0s into an array
            census_array = (I[min_y:max_y, min_x:max_x] < I[y][x]).flatten()
            
            # Convert this array into an int
            census_int = census_array.dot(1 << np.arange(census_array.size)[::-1])
            
            I_ct[y][x] = census_int
    
    return I_ct


def hamming_distance(xor_num):
    """
    Computes the hamming distance between two bitstrings given that they have 
    already been XOR'ed. Thus, the distance is simply the number of different 
    bits, which will be the number of 1s in the XOR'ed binary number

    Parameters:
    -----------
    xor_num (int): an integer that represents two numbers that have been XOR'ed

    Returns:
    --------
    hamming_dist (int): the hamming distance
    """
    
    # Converts the int to a binary number and counts the number of 1s
    bitstring = bin(xor_num)
    hamming_dist = bitstring.count('1')
    return hamming_dist


if __name__ == "__main__":
    # Load the stereo images
    It = imread('./input/run1_base_hr/omni_image4/frame000000_2018_09_04_17_19_42_773316.png', as_gray = True)
    Ib = imread('./input/run1_base_hr/omni_image5/frame000000_2018_09_04_17_19_42_773316.png', as_gray = True)

    # Camera intrinsic matrixes and distortion arrays
    Kt = np.array([[473.571, 0,  378.17],
          [0,  477.53, 212.577],
          [0,  0,  1]])
    Kb = np.array([[473.368, 0,  371.65],
          [0,  477.558, 204.79],
          [0,  0,  1]])
    dt = np.array([-0.333605,0.159377,6.11251e-05,4.90177e-05,-0.0460505])
    db = np.array([-0.3355,0.162877,4.34759e-05,2.72184e-05,-0.0472616])
    
    imageSize = (752, 480)
    T = np.array([0, 0.12, 0])

    # Rectifying images
    It_rect, Ib_rect = rectify_images(It, Ib, Kt, Kb, dt, db, imageSize, T)[0:2]

    # Calculate disparity
    Id = get_disparity(It_rect, Ib_rect)

    # Plotting rectified images and disparity
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax1.imshow(It_rect, cmap='gray')
    ax2.imshow(Ib_rect, cmap='gray')
    ax3.imshow(-Id, cmap='gray')
    plt.show()
