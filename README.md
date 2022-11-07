# Degree day based model to predict egg hatching of *Philaenus spumarius* the vector of *Xylella fastidiosa* in Europe

This repository contains some example codes with the necessary data used to predict egg hatching for *Philaenus spumarius*, published in.

A complete reproduction of the paper in [] requires advanced programming knowledge of different programming languages (Python, R, Julia) and database usage (Copernicus). If you are interested in further development and you have any doubt just feel free to contact me at alex@ifisc.uib-csic.es

# Example code usage

* Download temperature data from copernicus [ERA5-Land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview) service in grib format. Download a separate file for each year of interest. Alternatively, here we provide some files for an easier use:

* To get the example code working, move grib files to a folder named GRIB_files, or change the corresponding line of code to your preferred path.

* Run Example-GDD_computation_ERA5_Land notebook to compute daily accumulated GDD data.

* Run Example-Spain_predictions notebook to generate an animation of the accumulated probability of egg hatching in time for the given selected year.
