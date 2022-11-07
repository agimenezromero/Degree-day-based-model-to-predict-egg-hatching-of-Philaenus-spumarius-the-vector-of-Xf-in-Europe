import requests
import sys
import os

os.mkdir("GRIB_files")

names = ["Spain_2015.grib", "Spain_2016.grib", "Spain_2017.grib", "Spain_2018.grib", "Spain_2019.grib", "Spain_2020.grib", "Spain_2021.grib"]

for name in names:

    print("Downloading %s... " % name, end = "")

    url = 'https://ifisc.uib-csic.es/users/alex/GRIB_files/%s' % name

    r = requests.get(url, allow_redirects=True)

    open('GRIB_files/%s' % name, 'wb').write(r.content)

    print("Done!")
