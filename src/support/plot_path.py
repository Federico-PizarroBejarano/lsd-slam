import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

def plot_path(full_path = np.array([]), interpolated_path = np.array([]), calculated_path = np.array([])):
    overhead_file = './input/raster_data/mosaic_utm_20cm.tif'
    overhead = imread(overhead_file)

    extent = (625436.02725488, 625563.22725488, 5041709.408990768, 5041778.008990767)

    plt.imshow(overhead, extent=extent)

    if full_path.shape[0] > 0:
        plt.scatter(full_path[:,0], full_path[:,1], s=1, color='b')

    if interpolated_path.shape[0] > 0:
        plt.scatter(interpolated_path[:,0], interpolated_path[:,1], s=1, color='g')

        for i in range(interpolated_path.shape[0]):
            plt.annotate(i, (interpolated_path[i, 0], interpolated_path[i, 1]))
    
    if calculated_path.shape[0] > 0:
        plt.scatter(calculated_path[:,0], calculated_path[:,1], s=1, color='r')

        for i in range(calculated_path.shape[0]):
            plt.annotate(i, (calculated_path[i, 0], calculated_path[i, 1]))

    plt.show()
