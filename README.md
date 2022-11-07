# Degree day based model to predict egg hatching of *Philaenus spumarius* the vector of *Xylella fastidiosa* in Europe

This repository contains some example codes with the necessary data used to predict egg hatching for *Philaenus spumarius*, published in.

A complete reproduction of the paper in [] requires advanced programming knowledge of different programming languages (Python, R, Julia) and database usage (Copernicus). If you are interested in further development and you have any doubt just feel free to contact me at alex@ifisc.uib-csic.es

# Example code usage

## Requirements

Python 3.x installed with the following libraries

* Numpy
* Matplotlib
* Cartopy
* Datetime

Julia with GRIB.jl library installed

## Instructions

* Download the temperature data from https://cloud.ifisc.uib-csic.es/nextcloud/index.php/s/XEYCJieCNTyywkx or using the provided download_data.py script.

* If manually downloading the data from the nextcloud link, create a folder named GRIB_files into the current directory and move there the grib files (the download_data.py scripts already does this automatically for you).

* Run Example-GDD_computation_ERA5_Land notebook to compute daily accumulated GDD data.

* Run Example-Spain_predictions notebook to generate an animation of the accumulated probability of egg hatching in time for the given selected year.
