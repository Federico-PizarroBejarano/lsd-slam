import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

def plot_path(overhead_file, true_utm_poses, utm_poses = np.array([])):
    overhead = imread(overhead_file)

    extent = (625436.02725488, 625563.22725488, 5041709.408990768, 5041778.008990767)

    plt.imshow(overhead, extent=extent)
    plt.scatter(true_utm_poses[:,0], true_utm_poses[:,1], s=1, color='b')

    if utm_poses.shape[0] != 0:
        plt.scatter(utm_poses[:,0], utm_poses[:,1], s=1, color='r')

    plt.show()


if __name__ == "__main__":
    overhead_file = './input/raster_data/mosaic_utm_20cm.tif'
    gps_file = './input/run1_base_hr/gps-utm18t.txt'

    plot_path(overhead_file, gps_file)