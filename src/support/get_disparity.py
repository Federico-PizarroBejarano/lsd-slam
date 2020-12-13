import numpy as np
from scipy.ndimage.filters import correlate, median_filter

def get_disparity(Il, Ir, bbox, maxd):
    """
    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
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
 
    Id = np.zeros(Il.shape)
    
    # Set all parameters
        # window: window size for SAD
        # transform_size: window size for census transform 
        # median_window_size: size of median filter used at the end
    window = 16
    transform_size = 2
    median_window_size = 12
   
    height = Il.shape[0]
    width = Il.shape[1]
    
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
            # Confirm pixel is within bounding box
            if x > bbox[0, 1] or x < bbox[0, 0] or y > bbox[1, 1] or y < bbox[1, 0]:
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

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id


def census_transform(I, transform_size):
    """
    This function applies the census transform to an image. 

    Parameters:
    -----------
    I               - An image, m x n pixel np.array, greyscale.
    transform_size  - The window size of the transform

    Returns:
    --------
    I_ct            - An m x n np.array representing the census-transformed image
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
    xor_num   - A Python int that represents two numbers that have been XOR'ed

    Returns:
    --------
    hamming_dist  - An integer of the hamming distance
    """
    
    # Converts the int to a binary number and counts the number of 1s
    bitstring = bin(xor_num)
    hamming_dist = bitstring.count('1')
    return hamming_dist