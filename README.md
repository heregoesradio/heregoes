# heregoes

<p>
<a href="https://github.com/heregoesradio/heregoes/blob/main/LICENSE.txt"><img alt="License: GPL-3.0" src="https://img.shields.io/github/license/heregoesradio/heregoes"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This is the core processing library used by [Here GOES Radiotelescope (2020)](https://heregoesradio.com/), a sculptural ground station for the GRB downlink of the NOAA weather satellite GOES-16. The library renders basic ABI and SUVI imagery from netCDF, and offers some useful functions for navigating and projecting ABI data. Some additional convenience functions for working with 2D NumPy arrays are provided in [util](heregoes/util/__init__.py). To optimize for the real-time processing needs of Here GOES Radiotelescope, array functions are accelerated using [Numba](https://numba.pydata.org/) with parallelism controlled by environmental variables.

<p float="left">
    <a href="https://static.heregoesradio.com/abi/fulldisk/2019-09-04/grb_g16_fulldisk_color_2019-09-04T170015Z.jpg">
        <img src="https://static.heregoesradio.com/abi/fulldisk/2019-09-04/grb_g16_fulldisk_color_2019-09-04T170015Z_thumbnail.jpg" width="250">
    </a>
    <a href="http://static.heregoesradio.com/abi/fulldisk/2019-09-04/grb_g16_fulldisk_color_2019-09-04T170015Z_cog.jpg">
        <img src="http://static.heregoesradio.com/abi/fulldisk/2019-09-04/grb_g16_fulldisk_color_2019-09-04T170015Z_cog_thumbnail.jpg" height="250">
    </a>
</p>

---

## Setup

Clone this repository and install the Conda environment. For Intel machines, use heregoes-env-intel.yml which uses MKL for acceleration. For other architectures, including ARM64 (e.g. Raspberry Pi 4), use heregoes-env-other.yml which installs with OpenBLAS.
```
conda env create -f heregoes-env-intel.yml
conda activate heregoes-env
```

Optional environmental variables:
- `HEREGOES_ENV_PARALLEL`: Defaults to `False`
- `HEREGOES_ENV_NUM_CPUS`: Number of CPUs to use if `HEREGOES_ENV_PARALLEL` is `True`. Defaults to the number of CPUs reported by the OS
- `HEREGOES_ENV_IREMIS_DIR`: Directory path of the UW CIMSS IREMIS dataset which can be downloaded [here](https://cimss.ssec.wisc.edu/iremis/)

---

## Using the NCInterface

The `load` function in the heregoes library takes an ABI or SUVI netCDF file and returns either `ABIObject` or `SUVIObject` extensions of the `NCInterface` to browse and interact with the netCDF contents. It mimicks the structure and behavior of a netCDF4 `Dataset` object, except small variables are all stored in memory and larger variables are lazy-loaded from disk when accessed with an index.

### Retrieve radiance metadata for an ABI L1b netCDF

With an ABI L1b radiance netCDF file `abi_nc`, print netCDF data for the Rad variable:

```python
from heregoes import load
abi_data = load(abi_nc)
print(abi_data['Rad'])
```

Which returns:
```
<class 'heregoes.ncinterface._NCVar'>
int16 Rad(y, x)
    var_name: Rad
    long_name: ABI L1b Radiances
    standard_name: toa_outgoing_radiance_per_unit_wavelength
    sensor_band_bit_depth: 12
    valid_range: [   0 4094]
    scale_factor: 0.15859237
    add_offset: -20.289911
    units: W m-2 sr-1 um-1
    resolution: y: 0.000014 rad x: 0.000014 rad
    coordinates: band_id band_wavelength t y x
    grid_mapping: goes_imager_projection
    cell_methods: t: point area: point
    ancillary_variables: DQF
    ndim: 2
    shape: (6000, 10000)
    size: 60000000
    dtype: int16
    dimensions: (<class 'heregoes.ncinterface._NCDim'>: name = 'y', size = 6000, <class 'heregoes.ncinterface._NCDim'>: name = 'x', size = 10000)
```

### Check the exposure time of a SUVI L1b netCDF

With an L1b SUVI netCDF file `suvi_nc`, view the exposure time in seconds:

```python
from heregoes import load
suvi_data = load(suvi_nc)
print(suvi_data["CMD_EXP"][:])
```

Which returns:
```
[1.]
```

---

## Imagery Examples

The heregoes library renders standard imagery for the GOES-R ABI and SUVI instruments following available literature wherever possible.

### Render a single-channel SUVI image

SUVI L1b netCDF files with 1-second exposure times can be rendered into images:

```python
from heregoes import image

suvi_image = image.SUVIImage(suvi_nc)
suvi_image.save('suvi.jpg')
```
<img src="example-images/suvi-small.jpg" width="500">

### Render a single-channel ABI image

With an ABI L1b radiance netCDF file `abi_nc`, render the L2 CMI product:

```python
from heregoes import image

abi_image = image.ABIImage(abi_nc, gamma=0.75)
abi_image.save('abi.jpg')
```
<img src="example-images/abi-small.jpg" width="500">

### Render the ABI 'natural' color RGB

With ABI L1b radiance files for the 0.64 μm (`red_nc`), 0.86 μm (`green_nc`), and 0.47 μm (`blue_nc`) components:

```python
from heregoes import image

abi_rgb = image.ABINaturalRGB(red_nc, green_nc, blue_nc, gamma=0.75)
abi_rgb.save('rgb.jpg')
```
<img src="example-images/rgb-small.jpg" width="500">

### Create a Cloud-Optimized GeoTIFF from an ABI RGB

With the `abi_rgb` object from the previous step:

```python
from heregoes import projection

abi_projection = projection.ABIProjection(abi_rgb.abi_data)
abi_projection.resample2cog(abi_rgb.bv, 'rgb.tiff')
```
<img src="example-images/rgb-cog-small.jpg" width="500">

---

## Navigation Examples

Navigation routines for ABI return useful geometry on the ABI Fixed Grid: pixel latitude and longitude, pixel area, and look angles for Sun and satellite.

### Navigate an ABI image to latitude and longitude

With an ABI L1b radiance netCDF file `abi_nc`:

```python
from heregoes import load, navigation

abi_data = load(abi_nc)
abi_navigation = navigation.ABINavigation(abi_data)
abi_navigation.lat_deg #latitude array
abi_navigation.lon_deg #longitude array
```

### Calculate the per-pixel spatial coverage of an ABI image

With the `abi_navigation` object from the previous step:

```python
abi_navigation.area_m #area of each pixel in square meters
```

### Calculate per-pixel Sun and satellite vector angles of an ABI image

With the `abi_navigation` object from the previous step:

```python
abi_navigation.sun_za #solar zenith angle
abi_navigation.sun_az #solar azimuth in North-clockwise convention

abi_navigation.sat_za #satellite zenith angle
abi_navigation.sat_az #satellite azimuth in North-clockwise convention
```

### Find a pixel in an ABI image from latitude and longitude

With an ABI L1b radiance netCDF file `abi_nc`:

```python
from heregoes import load, navigation

abi_data = load(abi_nc)
abi_navigation = navigation.ABINavigation(abi_data, lat_deg=44.72609499, lon_deg=-93.02279070)
abi_navigation.index #nearest (y, x) index of the ABI image at provided latitude and longitude
```

---

## Ancillary Dataset Examples

The heregoes library can project georeferenced datasets from WGS84 to the ABI Fixed Grid and vice versa.

### Create a GSHHS/Natural Earth boolean water mask projected to an ABI image

With an ABI L1b radiance netCDF file `abi_nc`:

```python
import cv2
from heregoes import ancillary, load

abi_data = load(abi_nc)
water = ancillary.WaterMask(abi_data, rivers=True)
cv2.imwrite('water.jpg', water.data['water_mask'] * 255)
```
<img src="example-images/water-small.jpg" width="500">

### Load IREMIS land emissivity and project it for an ABI channel 7 image

With an ABI L1b radiance netCDF file for 3.9 μm `c07_nc`:

```python
import cv2
from heregoes import ancillary, load, util

c07_data = load(c07_nc)
iremis = ancillary.IREMIS(c07_data)
cv2.imwrite('iremis.jpg', util.minmax(iremis.data['c07_land_emissivity']) * 255)
```
<img src="example-images/iremis-small.jpg" width="500">