# Copyright (c) 2020-2023.

# Author(s):

#   Harry Dove-Robinson <admin@wx-star.com>
#   for Here GOES Radiotelescope (Harry Dove-Robinson & Heidi Neilson)

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

import os
import uuid

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

from heregoes import NUM_CPUS

GDAL_PARALLEL = False
if os.getenv("HEREGOES_ENV_PARALLEL", "False").lower() == "true":
    GDAL_PARALLEL = True


class ABIProjection:
    """
    This is a class for projecting NumPy arrays to and from the ABI fixed grid in memory with GDAL.

    Arguments:
        - `abi_data`: The ABIObject formed on a GOES-R ABI L1b Radiance netCDF file as returned by `heregoes.load()`

    Class methods:
        - `resample2abi(latlon_array)` resamples an array with WGS84 lat/lon projection to the ABI fixed grid domain. Returns the resampled array if convert_np is `True` (default), otherwise returns the GDAL dataset
        - `resample2latlon(abi_array)` resamples an ABI array from the ABI fixed grid domain to WGS84 lat/lon projection. Returns the resampled array if convert_np is `True` (default), otherwise returns the GDAL dataset
        - `resample2cog(abi_array, cog_filepath)` resamples an ABI array from the ABI fixed grid domain to WGS84 lat/lon projection and saves to a Cloud Optimized GeoTIFF (COG) at the filepath `cog_filepath`
    """

    def __init__(self, abi_data):
        self.abi_data = abi_data
        h = self.abi_data["goes_imager_projection"].perspective_point_height
        a = self.abi_data["goes_imager_projection"].semi_major_axis
        b = self.abi_data["goes_imager_projection"].semi_minor_axis
        f = 1 / self.abi_data["goes_imager_projection"].inverse_flattening
        lat_0 = self.abi_data["goes_imager_projection"].latitude_of_projection_origin
        lon_0 = self.abi_data["goes_imager_projection"].longitude_of_projection_origin
        sweep = self.abi_data["goes_imager_projection"].sweep_angle_axis

        ul_x = (self.abi_data["x_image_bounds"][0] * h).item()
        ul_y = (self.abi_data["y_image_bounds"][0] * h).item()
        lr_x = (self.abi_data["x_image_bounds"][1] * h).item()
        lr_y = (self.abi_data["y_image_bounds"][1] * h).item()
        self.abi_bounds = [ul_x, ul_y, lr_x, lr_y]

        self._intermediate_format = "GTiff"
        self._intermediate_gdal_options = ["COMPRESS=NONE", f"NUM_THREADS={NUM_CPUS}"]

        if self.abi_data.band_id_safe == "Color":
            self._intermediate_gdal_options += ["PHOTOMETRIC=RGB"]

        self.latlon_srs = "+proj=latlon +ellps=WGS84 +datum=WGS84 +no_defs"
        self.abi_srs = f"+proj=geos +h={h} +a={a} +b={b} +f={f} +lat_0={lat_0} +lon_0={lon_0} +x_0=0.0 y_0=0.0 +sweep={sweep} +ellps=GRS80 +no_defs"

        # maps NumPy to GDAL data types
        self._np_gdal_typemap = {
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

    def _make_dataset(self, array, reverse_channels=True):
        # creates a GDAL dataset to store an n-d NumPy array in memory.
        # if the array has more than 2 dimensions, each dimension is stored as a GDAL raster band
        # by default we reverse the order of band loading to reconcile OpenCV using BGR and GDAL using RGB

        if array.ndim > 2:
            bands = array.ndim
        else:
            bands = 1

        drv = gdal.GetDriverByName("MEM")
        ds = drv.Create(
            "",
            xsize=array.shape[1],
            ysize=array.shape[0],
            bands=bands,
            eType=self._np_gdal_typemap[array.dtype.name],
        )

        if reverse_channels:
            band_range = reversed(range(0, bands))
        else:
            band_range = range(0, bands)

        for i in band_range:
            slc = [slice(None)] * bands
            if bands > 1:
                slc[-1] = i
            ds.GetRasterBand(bands - i).WriteArray(array[tuple(slc)])
            ds.GetRasterBand(bands - i).FlushCache()

        return ds

    def _gdal2numpy(self, ds, reverse_channels=True):
        # by default we reverse the order of band loading to reconcile OpenCV using BGR and GDAL using RGB
        gdal_bands = ds.RasterCount

        if gdal_bands > 1:
            if reverse_channels:
                band_range = reversed(range(0, gdal_bands))
            else:
                band_range = range(0, gdal_bands)

            np_image = []
            for i in band_range:
                np_image.append(np.array(ds.GetRasterBand(i + 1).ReadAsArray()))

            np_image = np.stack(np_image, axis=2)

        else:
            np_image = np.array(ds.GetRasterBand(1).ReadAsArray())

        return np_image

    def _resample(self, ds, translate_options, warp_options):
        # use a UUID so that in-memory rasters are unique for multiprocessing
        unique = uuid.uuid4()
        gdal_mem_translate = f"/vsimem/gdal_mem_translate_{unique}.tif"
        gdal_mem_warp = f"/vsimem/gdal_mem_warp_{unique}.tif"

        # georeference
        gdal.Translate(gdal_mem_translate, ds, options=translate_options)
        del ds

        # project
        gdal.Warp(gdal_mem_warp, gdal_mem_translate, options=warp_options)
        gdal.Unlink(gdal_mem_translate)
        resampled = gdal.Open(gdal_mem_warp, 1)
        gdal.Unlink(gdal_mem_warp)

        return resampled

    def resample2abi(
        self,
        latlon_array,
        latlon_bounds=[-180.0, 90.0, 180.0, -90.0],
        interpolation="nearest",
        convert_np=True,
    ):
        # GDAL bounds are ul_x, ul_y, lr_x, lr_y
        ds = self._make_dataset(latlon_array)

        translate_options = gdal.TranslateOptions(
            outputSRS=self.latlon_srs,
            outputBounds=latlon_bounds,
            format=self._intermediate_format,
            resampleAlg=interpolation.lower(),
            creationOptions=self._intermediate_gdal_options,
        )
        warp_options = gdal.WarpOptions(
            srcSRS=self.latlon_srs,
            dstSRS=self.abi_srs,
            outputBounds=self.abi_bounds,
            width=self.abi_data.dimensions["x"].size,
            height=self.abi_data.dimensions["y"].size,
            format=self._intermediate_format,
            resampleAlg=interpolation.lower(),
            creationOptions=self._intermediate_gdal_options,
            multithread=GDAL_PARALLEL,
        )

        resampled = self._resample(ds, translate_options, warp_options)
        del ds

        if convert_np:
            resampled = np.flipud(self._gdal2numpy(resampled))

        return resampled

    def resample2latlon(self, abi_array, interpolation="bilinear", convert_np=True):
        ds = self._make_dataset(abi_array)

        translate_options = gdal.TranslateOptions(
            outputSRS=self.abi_srs,
            outputBounds=self.abi_bounds,
            format=self._intermediate_format,
            resampleAlg=interpolation.lower(),
            creationOptions=self._intermediate_gdal_options,
        )
        warp_options = gdal.WarpOptions(
            srcSRS=self.abi_srs,
            dstSRS=self.latlon_srs,
            format=self._intermediate_format,
            resampleAlg=interpolation.lower(),
            creationOptions=self._intermediate_gdal_options,
            multithread=GDAL_PARALLEL,
        )

        resampled = self._resample(ds, translate_options, warp_options)
        del ds

        if convert_np:
            resampled = self._gdal2numpy(resampled)

        return resampled

    def resample2cog(
        self,
        abi_array,
        cog_filepath,
        interpolation="LANCZOS",
        gdal_compression_algo="LZW",
    ):
        resampled = self.resample2latlon(
            abi_array, interpolation=interpolation, convert_np=False
        )

        final_gdal_options = [
            f"COMPRESS={gdal_compression_algo}",
            f"NUM_THREADS={NUM_CPUS}",
            f"PREDICTOR=YES",
            f"OVERVIEW_RESAMPLING={interpolation}",
        ]

        drv = gdal.GetDriverByName("COG")
        drv.CreateCopy(str(cog_filepath), resampled, options=final_gdal_options)

        return cog_filepath
