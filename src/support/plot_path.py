import rasterio
import rasterio.plot
import numpy as np
import matplotlib.pyplot as plt


def plot_path(overhead_file, gps_file):
    overhead = rasterio.open(overhead_file)
    gps = np.genfromtxt(gps_file, delimiter=",")

    mosaic_rgb = np.dstack(tuple(overhead.read(i) for i in [1,2,3]))
    extent = rasterio.plot.plotting_extent(overhead)

    plt.imshow(mosaic_rgb, extent=extent)
    plt.scatter(gps[:,1], gps[:,2], s=1, color='b')

    plt.show()


if __name__ == "__main__":
    overhead_file = './input/raster_data/mosaic_utm_20cm.tif'
    gps_file = './input/run1_base_hr/gps-utm18t.txt'

    plot_path(overhead_file, gps_file)