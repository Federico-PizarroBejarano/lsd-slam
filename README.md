# Visual Odometry for CPET Dataset
This is my final project for `ROB501: Computer Vision for Robotics`. Here I create a simple visual odometry pipeline based on Large-Scale Direct Stereo SLAM ([LSD-SLAM](https://vision.in.tum.de/research/vslam/lsdslam))[1] to localize a rover on Canadian Planetary Emulation Terrain ([CPET](https://starslab.ca/enav-planetary-dataset/)) dataset. There is no mapping, simply localization, and thus the localization estimate tends to drift significantly. However, it is a proof of concept for semi-dense depth map based visual odometry using stereo cameras. 

## Functionality
Using stereo images from one of the rover's runs across the Martian-like terrain, disparity images are generated using the `get_disparity.py` function, a disparity map generator based on [census transforms](http://www.cs.cornell.edu/~rdz/Papers/ZW-ECCV94.pdf) [2]. Using non-linear optimization, the homogenous transformation from one position to the next are calculated. The UTM position of the rover at each increment is found using these transforms and plotted onto an overhead image of the terrain. 

## Results
Read my final report in `Dense Visual Odometry.pdf`. 

## Install
1. git clone this repository and create an input and an output folder beside the `src` folder. Call them `input` and `output`.
2. Download `raster_data`, `run1_base_hr/omni_image4`, and `run1_base_hr/omni_image5` from the CPET dataset above and put the folders in the `input` folder. Do not change the name of the files or folders. 

Your folder structure should look like:

```
input/
    run1_base_hr/
        omni_image4/
        omni_image5/
    raster_data/
output/
src/
```
3. Install docker desktop

## Run
1. Simply run `docker-compose up` (or `sudo docker-compose up`) which will pull the correct docker container and run the code. This will generate the disparity images of all images from `start_frame` to `end_frame` and plot the path on the overhead image. You can change these two variables in `rob501_project.py`. 
