# LSD-Slam for CPET Dataset
This is my final project for ROB501: Computer Vision for Robotics. Here I recreate a simplified version of Large-Scale Direct Stereo SLAM (LSD-SLAM, found [here](https://vision.in.tum.de/research/vslam/lsdslam)) to localize a rover on Canadian Planetary Emulation Terrain (CPET) dataset (found [here](https://starslab.ca/enav-planetary-dataset/)). 

## Functionality
Using stereo images from one of the rover's runs across the Martian-like terrain, disparity images are generated using the `get_disparity.py` function. These disparity images are used to create a disparity map of the terrain. Using non-linear optimization, UTM positions of the rover at each time increment are calculated and plotted onto an overhead image of the terrain. 


## Install
1. TBD

## Run
1. TBD
