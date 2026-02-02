# Copyright (c) 2020-2025.

# Author(s):

#   R. Dove <admin@wx-star.com>
#   An early version of heregoes runs at Here GOES Radiotelescope
#   (Dove & Neilson, 2020) <heregoesradio.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import uuid
from pathlib import Path

import numpy as np
from osgeo import gdal

from heregoes.core import NUM_CPUS


def gdal2cog(
    gdal_dataset, filepath, compression_algo="lzw", overview_resampling_algo="lanczos"
):
    final_gdal_options = [
        f"COMPRESS={compression_algo}",
        f"NUM_THREADS={NUM_CPUS}",
        f"PREDICTOR=YES",
        f"OVERVIEW_RESAMPLING={overview_resampling_algo}",
    ]

    filepath = Path(filepath)
    if filepath.exists() and (filepath.is_file() or filepath.is_symlink()):
        filepath.unlink()

    drv = gdal.GetDriverByName("COG")
    drv.CreateCopy(str(filepath), gdal_dataset, options=final_gdal_options)

    return filepath


def translate_and_warp(gdal_dataset, translate_options, warp_options):
    # use a UUID so that in-memory rasters are unique for multiprocessing
    unique = uuid.uuid4()
    gdal_mem_translate = f"/vsimem/gdal_mem_translate_{unique}.tif"
    gdal_mem_warp = f"/vsimem/gdal_mem_warp_{unique}.tif"

    # georeference
    gdal.Translate(gdal_mem_translate, gdal_dataset, options=translate_options)
    del gdal_dataset  # is this really necessary? many examples have it supposedly to save memory IF garbage collection occurs before this function exits

    # project
    gdal.Warp(gdal_mem_warp, gdal_mem_translate, options=warp_options)
    gdal.Unlink(gdal_mem_translate)
    resampled = gdal.Open(gdal_mem_warp, 1)
    gdal.Unlink(gdal_mem_warp)

    return resampled


def numpy2gdal(array, reverse_channels=True):
    # creates a GDAL dataset to store an n-d NumPy array in memory.
    # if the array has more than 2 dimensions, each dimension is stored as a GDAL raster band
    # by default we reverse the order of band loading to reconcile OpenCV using BGR and GDAL using RGB

    if array.ndim > 2:
        bands = array.ndim
    else:
        bands = 1

    # maps NumPy to GDAL data types
    np_gdal_typemap = {
        "int8": 1,
        "complex128": 11,
        "complex64": 10,
        "float64": 7,
        "float32": 6,
        "int16": 3,
        "int32": 5,
        "uint8": 1,
        "uint16": 2,
        "uint32": 4,
    }

    drv = gdal.GetDriverByName("MEM")
    gdal_dataset = drv.Create(
        "",
        xsize=array.shape[1],
        ysize=array.shape[0],
        bands=bands,
        eType=np_gdal_typemap[array.dtype.name],
    )

    if reverse_channels:
        band_range = reversed(range(0, bands))
    else:
        band_range = range(0, bands)

    for i in band_range:
        slc = [slice(None)] * bands
        if bands > 1:
            slc[-1] = i
        gdal_dataset.GetRasterBand(bands - i).WriteArray(array[tuple(slc)])
        gdal_dataset.GetRasterBand(bands - i).FlushCache()

    return gdal_dataset


def gdal2numpy(gdal_dataset, reverse_channels=True):
    # by default we reverse the order of band loading to reconcile OpenCV using BGR and GDAL using RGB
    gdal_bands = gdal_dataset.RasterCount

    if gdal_bands > 1:
        if reverse_channels:
            band_range = reversed(range(0, gdal_bands))
        else:
            band_range = range(0, gdal_bands)

        np_image = []
        for i in band_range:
            np_image.append(np.array(gdal_dataset.GetRasterBand(i + 1).ReadAsArray()))

        np_image = np.stack(np_image, axis=2)

    else:
        np_image = np.array(gdal_dataset.GetRasterBand(1).ReadAsArray())

    return np_image
