# heregoes
## Lightweight Python for GOES-R ABI and SUVI
<p>
<a href="https://github.com/heregoesradio/heregoes/blob/main/LICENSE.txt"><img alt="License: GPL-3.0" src="https://img.shields.io/github/license/heregoesradio/heregoes"></a>
<a href="./coverage/coverage.xml"><img src="./coverage/coverage-badge.svg?dummy=8484744"></a>
<a href="https://zenodo.org/badge/latestdoi/469245509"><img src="https://zenodo.org/badge/469245509.svg"></a>
<p>

------------------------------------------

<p float="left">
    <a href="https://static.heregoesradio.com/abi/fulldisk/2019-09-04/g16_abi_fulldisk_color_2019-09-04T170015Z_cog_v1.0.1.jpg">
        <img src="https://static.heregoesradio.com/abi/fulldisk/2019-09-04/g16_fulldisk_color_2019-09-04T170015Z_cog_thumbnail.jpg" height="250">
    </a>
    <a href="https://static.heregoesradio.com/suvi/grb_g16_suvi_color_2022-11-23T113653Z.jpg">
        <img src="https://static.heregoesradio.com/suvi/grb_g16_suvi_color_2022-11-23T113653Z_thumbnail.jpg" height="250">
    </a>
</p>

### Research quality GOES-R Earth and Sun imagery from netCDF

- Originally developed for realtime GOES-R processing at [Here GOES Radiotelescope](https://heregoesradio.com/) (Dove & Neilson, 2020)
- ABI features [tested](https://github.com/heregoesradio/heregoes/tree/main/tests) against ground targets and official data and literature
- SUVI imagery in production at [UW–Madison SSEC](https://cimss.ssec.wisc.edu/satellite-blog/archives/53279) and tested for QC
- Accelerated and parallelized with the [Numba](https://numba.pydata.org/) JIT compiler

## Features
| Instrument | Products | Features|
|:-----|----------|---------------------------------------------------------------|
| ABI  | L1b      | Render Cloud and Moisture Imagery and "Natural" color RGB     |
| ABI  | L1b, L2+ | Lat/lon and Fixed Grid subsetting with parallax correction    |
| ABI  | L1b, L2+ | Pixelwise navigation, ground coverage, and look vectors       |
| ABI  | L1b, L2+ | Resample Numpy arrays to and from the projection of ABI scenes|
| SUVI | L1b      | Extreme Ultraviolet solar imagery (long exposures)            |

## Documentation
- Imagery examples for [ABI](https://github.com/heregoesradio/heregoes/blob/main/heregoes/image/ABI.md) and [SUVI](https://github.com/heregoesradio/heregoes/blob/main/heregoes/image/SUVI.md)
- [ABI navigation, subsetting, and parallax correction](https://github.com/heregoesradio/heregoes/blob/main/heregoes/navigation/README.md)
- [Library reference](https://docs.heregoesradio.com)

## Demos
- [Terrain correction of the ABI Fixed Grid using heregoes](https://github.com/heregoesradio/heregoes/blob/main/demo/README.md)

## Quickstart
### 1. Install heregoes-env
Clone this repository and install the appropriate Conda environment for your CPU:

##### Intel (MKL)
```
conda env create -f release/heregoes-env-intel.yml
```

##### AMD, ARM64 (OpenBLAS)
```
conda env create -f release/heregoes-env-other.yml
```

### 2. Activate
```
conda activate heregoes-env
```

### 3. Environmental variables
Optionally set `HEREGOES_ENV_PARALLEL=False` to disable parallel execution,
or set `HEREGOES_ENV_NUM_CPUS=n` to limit the CPUs used to `n`.

### 4. netCDF input
Provide GOES-R ABI or SUVI netCDF files to `heregoes` from [NOAA CLASS](https://www.class.noaa.gov), [AWS S3](https://noaa-goes19.s3.amazonaws.com/index.html), or in real time from [CSPP Geo GRB](https://cimss.ssec.wisc.edu/csppgeo/grb.html).

```python
from heregoes.image import ABIImage, SUVIImage

abi_img = ABIImage("OR_ABI-L1b-Rad[...].nc")
abi_img.save("abi.jpg")

suvi_img = SUVIImage("OR_SUVI-L1b-[...].nc")
suvi_img.save("suvi.png")
```

## Planned
- ABI pixelwise timestamps
- Builds for conda-forge

## Future
- Dask integration
- Support for GeoXO