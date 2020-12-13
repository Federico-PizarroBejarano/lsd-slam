import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

def plot_path(overhead_file, gps_file):
    overhead = imread(overhead_file)
    gps = np.genfromtxt(gps_file, delimiter=",")

    extent = (625436.02725488, 625563.22725488, 5041709.408990768, 5041778.008990767)

    plt.imshow(overhead, extent=extent)
    plt.scatter(gps[:,1], gps[:,2], s=1, color='b')

    plt.show()


if __name__ == "__main__":
    overhead_file = './input/raster_data/mosaic_utm_20cm.tif'
    gps_file = './input/run1_base_hr/gps-utm18t.txt'

    plot_path(overhead_file, gps_file)