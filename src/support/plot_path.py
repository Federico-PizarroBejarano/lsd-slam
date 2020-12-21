import numpy as np
import matplotlib.pyplot as plt

def plot_path(input_dir, output_dir, full_path = np.array([]), interpolated_path = np.array([]), calculated_path = np.array([])):
    """
    Plots paths of the rover onto the overhead image. The paths must be supplied as numpy arrays of 
    utm coordinates. 

    Parameters
    ----------
    input_dir (str): the path to the input directory
    output_dir (str): the path to the output directory
    full_path (np.ndarray, optional): a numpy array where each row contains the utm coordinates of the rover.
        Represents the entire path of the rover using ground truth. Plotted in blue. 
    interpolated_path (np.ndarray, optional): a numpy array where each row contains the utm coordinates of the rover.
        Represents the interpolated path of the rover at the timestamps of the frames used in the calculation.
        Plotted in green.
    calculated_path (np.ndarray, optional): a numpy array where each row contains the utm coordinates of the rover.
        Represents the calculated path of the rover. Plotted in Red.
    """

    overhead_file = f'{input_dir}/raster_data/mosaic_utm_20cm.tif'
    overhead = plt.imread(overhead_file)

    extent = (625436.02725488, 625563.22725488, 5041709.408990768, 5041778.008990767)

    plt.imshow(overhead, extent=extent)

    if full_path.shape[0] > 0:
        plt.scatter(full_path[:,0], full_path[:,1], s=1, color='b', label='Full Path')

    if interpolated_path.shape[0] > 0:
        plt.scatter(interpolated_path[:,0], interpolated_path[:,1], s=1, color='g', label='True Path')
    
    if calculated_path.shape[0] > 0:
        plt.scatter(calculated_path[:,0], calculated_path[:,1], s=1, color='r', label='Calculated Path')

    plt.legend()
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.savefig(f'{output_dir}/final_path')

    plt.xlim(calculated_path[0,0]-6, calculated_path[0,0]+6)
    plt.ylim(calculated_path[0,1]-6, calculated_path[0,1]+6)

    plt.savefig(f'{output_dir}/final_path_zoomed')