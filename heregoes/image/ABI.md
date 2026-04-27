## ABI Examples

Create ABI radiance imagery with `ABIImage` and `ABINaturalRGB`, then save in common 8-bit formats with `.save()` or as Cloud-Optimized GeoTIFF (COG) with `.resample2cog()`. ABI image classes inherit subsetting, navigation, and parallax correction features from [`ABINavigation`](../navigation/README.md).

### ABI Cloud and Moisture Imagery from L1b Radiance
Generate the Cloud and Moisture Imagery (CMI) product following the GOES-R ATBD[^1] with `ABIImage`. Spectral radiance `.rad` is converted to CMI in `.cmi`, which contains either top-of-atmosphere reflectance factor for ABI bands 1-6, or brightness temperature in Kelvin for the emissive bands 7-16.
```python
import numpy as np

from heregoes.image import ABIImage

#optionally subset with a 2D `index` slice
y1, y2 = 0, 2000
x1, x2 = 0, 2000
index = np.s_[y1:y2, x1:x2]

#render the 0.64 µm image with square root enhancement and black space background
img = ABIImage(
    "OR_ABI-L1b-RadC-M6C02_G16_s20211691941174_e20211691943547_c20211691943571.nc",
    gamma=0.5,
    black_space=True,
    index=index,
)

#save as JPEG with a sequential filename in the current folder
img.save(ext=".jpeg")

#PosixPath('g16_abi_conus_c02_2021-06-18T194117Z.jpeg')
```

<a href="https://static.heregoesradio.com/abi/conus/g16_abi_conus_c02_2021-06-18T194117Z.jpeg">
<img src="https://static.heregoesradio.com/abi/conus/g16_abi_conus_c02_2021-06-18T194117Z_thumbnail.jpeg">
</a>

<br>

### ABI Natural Color RGB
Generate the "natural" color RGB for ABI using the fractional combination green band method[^2]:
```python
from heregoes.image import ABINaturalRGB

#optionally subset with a geographic bounding box
lat_bounds = [upper_left_lat, lower_right_lat]
lon_bounds = [upper_left_lon, lower_right_lon]

#optionally set gamma to 3/4 and scale up green and blue to the spatial resolution of the red channel
img = ABINaturalRGB(
    "OR_ABI-L1b-RadC-M6C02[...].nc",
    "OR_ABI-L1b-RadC-M6C03[...].nc",
    "OR_ABI-L1b-RadC-M6C01[...].nc",
    gamma=0.75,
    upscale=True,
    lat_bounds=lat_bounds,
    lon_bounds=lon_bounds,
)

#save to a JPEG
img.save(filepath="path/to/images/conus.jpg")

#or PNG (slower)
img.save(filepath="path/to/images/conus.png")
```

<a href="https://static.heregoesradio.com/abi/conus/g16_abi_conus_color_2019-09-04T170111Z.jpeg">
<img src="https://static.heregoesradio.com/abi/conus/g16_abi_conus_color_2019-09-04T170111Z_thumbnail.jpeg">
</a>

<br>

### Resample ABI imagery to equirectangular projection
```python
from heregoes.image import ABIImage

img = ABIImage("OR_ABI-L1b-RadM1-M6C13[...].nc")

#resample with GDAL bilinear interpolation and return as a Numpy array
arr = img.resample2latlon(resample_algo="bilinear")

#or save as a Cloud-Optimized GeoTIFF (COG)
img.resample2cog("meso1_c13.tiff", resample_algo="lanczos")
```


<a href="https://static.heregoesradio.com/abi/meso/meso1_c13.jpeg">
<img src="https://static.heregoesradio.com/abi/meso/meso1_c13.jpeg" width="500" height="500">
</a>

<br>

### References
[^1]: https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Enterprise/ATBD_Enterprise_Cloud_and_Moisture_Imagery_Product_v4_2021-01-13.pdf
[^2]: https://doi.org/10.1029/2018EA000379