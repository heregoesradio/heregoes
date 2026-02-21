# heregoes
## Lightweight Python for GOES-R ABI and SUVI
------------------------------------------
<p>
<a href="https://github.com/heregoesradio/heregoes/blob/main/LICENSE.txt"><img alt="License: GPL-3.0" src="https://img.shields.io/github/license/heregoesradio/heregoes"></a>
<a href="./coverage/coverage.xml"><img src="./coverage/coverage-badge.svg?dummy=8484744"></a>
<a href="https://zenodo.org/badge/latestdoi/469245509"><img src="https://zenodo.org/badge/469245509.svg"></a>
<p>

<p float="left">
    <a href="https://static.heregoesradio.com/abi/fulldisk/2019-09-04/grb_g16_fulldisk_color_2019-09-04T170015Z.jpg">
        <img src="https://static.heregoesradio.com/abi/fulldisk/2019-09-04/grb_g16_fulldisk_color_2019-09-04T170015Z_thumbnail.jpg" width="250">
    </a>
    <a href="http://static.heregoesradio.com/suvi/grb_g16_suvi_color_2022-11-23T113653Z.jpg">
        <img src="http://static.heregoesradio.com/suvi/grb_g16_suvi_color_2022-11-23T113653Z_thumbnail.jpg" height="250">
    </a>
</p>

### Research quality GOES-R Earth and Sun imagery from netCDF

- Purpose-built for realtime GOES-R processing at [Here GOES Radiotelescope](https://heregoesradio.com/) (Dove & Neilson, 2020)
- ABI features [tested](./tests/) against ground targets and official data and literature
- SUVI imagery in production at [UW–Madison SSEC](https://cimss.ssec.wisc.edu/satellite-blog/archives/53279) and tested for QC
- Accelerated and parallelized with the [Numba](https://numba.pydata.org/) JIT compiler

## Features
| Instrument | Products | Features|
|:-----|----------|---------------------------------------------------------------|
| ABI  | L1b      | Render Cloud Moisture Imagery and "Natural" color RGB         |
| ABI  | L1b, L2+ | Lat/lon and Fixed Grid subsetting with parallax correction    |
| ABI  | L1b, L2+ | Pixelwise navigation, ground coverage, and look vectors       |
| ABI  | L1b, L2+ | Resample Numpy arrays to and from the projection of ABI scenes|
| SUVI | L1b      | Extreme Ultraviolet solar imagery (long exposures)            |

## Documentation
- [ABI navigation, subsetting, and parallax correction](./heregoes/navigation/README.md)
- More soon!

## Demos
- [Terrain correction of the ABI Fixed Grid using heregoes](./demo/README.md)

## Planned
- [SUVI RGB support](https://static.heregoesradio.com/suvi/grb_g19_suvi_color_2026-02-02T050244Z.jpg)
- Builds for conda-forge

## Future
- Dask integration
- Support for GeoXO

## Quickstart
### Install
Clone this repository and install the Conda environment. For Intel machines, use `heregoes-env-intel.yml` which includes MKL for acceleration. For other architectures like AMD or ARM64 (e.g. Raspberry Pi 5), use `heregoes-env-other.yml` which installs with OpenBLAS. Not yet tested on Apple Silicon.

```
conda env create -f release/heregoes-env-intel.yml
conda activate heregoes-env
```


### Environmental variables
Set `HEREGOES_ENV_PARALLEL=False` to disable parallel execution,
or set `HEREGOES_ENV_NUM_CPUS=n` to limit the CPUs used to `n`.

### netCDF input
Provide GOES-R ABI or SUVI netCDF files to `heregoes` from [NOAA CLASS](https://www.class.noaa.gov), [AWS S3](https://noaa-goes19.s3.amazonaws.com/index.html), or in real time from [CSPP Geo GRB](https://cimss.ssec.wisc.edu/csppgeo/grb.html).

### ABI imagery from L1b radiance
```python
from heregoes.image import ABIImage, ABINaturalRGB

#render single-channel image
img = ABIImage("OR_ABI-L1b-RadC-M6C13[...].nc")

#or natural color RGB
img = ABINaturalRGB("OR_ABI-L1b-RadF-M6C02[...].nc", "OR_ABI-L1b-RadF-M6C03[...].nc", "OR_ABI-L1b-RadF-M6C01[...].nc", gamma=0.75)

#save to a JPEG
img.save("fulldisk.jpg")

#or as a Cloud-Optimized GeoTIFF in the plate carrée projection
img.resample2cog(source="bv", filepath="fulldisk.tiff")
```

### SUVI imagery
```python
from heregoes.image import SUVIImage

img = SUVIImage("OR_SUVI-L1b-[...].nc")
img.save("suvi.png")
```