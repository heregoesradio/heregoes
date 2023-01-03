# Copyright (c) 2020, 2021, 2022.

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

import uuid

import numpy as np
from osgeo import gdal

from heregoes import GDAL_PARALLEL, NUM_CPUS


class ABIProjection:
    """
    This is a class for projecting NumPy arrays to and from the ABI fixed grid in memory with GDAL.

    Arguments:
        - `abi_meta`: The NCMeta object formed on a GOES-R ABI L1b Radiance netCDF file

    Class methods:
        - `resample2abi(latlon_array)` resamples an array with WGS84 lat/lon projection to the ABI fixed grid domain. Returns the resampled array if convert_np is `True` (default), otherwise returns the GDAL dataset
        - `resample2latlon(abi_array)` resamples an ABI array from the ABI fixed grid domain to WGS84 lat/lon projection. Returns the resampled array if convert_np is `True` (default), otherwise returns the GDAL dataset
        - `resample2cog(abi_array, cog_filepath)` resamples an ABI array from the ABI fixed grid domain to WGS84 lat/lon projection and saves to a Cloud Optimized GeoTIFF (COG) at the filepath `cog_filepath`
    """

    def __init__(self, abi_meta):

        self.abi_meta = abi_meta
        h = self.abi_meta.instrument_meta.perspective_point_height
        a = self.abi_meta.instrument_meta.semi_major_axis
        b = self.abi_meta.instrument_meta.semi_minor_axis
        f = 1 / self.abi_meta.instrument_meta.inverse_flattening
        lat_0 = self.abi_meta.instrument_meta.latitude_of_projection_origin
        lon_0 = self.abi_meta.instrument_meta.longitude_of_projection_origin
        sweep = self.abi_meta.instrument_meta.sweep_angle_axis

        ul_x = self.abi_meta.instrument_meta.x_image_bounds[0] * h
        ul_y = self.abi_meta.instrument_meta.y_image_bounds[0] * h
        lr_x = self.abi_meta.instrument_meta.x_image_bounds[1] * h
        lr_y = self.abi_meta.instrument_meta.y_image_bounds[1] * h
        self.abi_bounds = [ul_x, ul_y, lr_x, lr_y]

        self._intermediate_format = "GTiff"
        self._intermediate_gdal_options = ["COMPRESS=NONE", f"NUM_THREADS={NUM_CPUS}"]
        # only used for file outputs
        self._final_gdal_options = []

        if self.abi_meta.instrument_meta.band_id == "Color":
            self._intermediate_gdal_options += ["PHOTOMETRIC=RGB"]
            self._final_gdal_options += ["PHOTOMETRIC=RGB"]

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

    def _cog_overviews(self, max_num=10, min_size=256):
        # creates a list of overviews up to `max_num` in length as small as `min_size`
        y = self.abi_meta.instrument_meta.y
        x = self.abi_meta.instrument_meta.x

        powers = [2]  # make at least 1 overview at half size
        for i in range(2, max_num + 1):
            power = 2 ** i
            if power <= max(y, x) / min_size:
                powers.append(power)

        return powers

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
            width=self.abi_meta.instrument_meta.x,
            height=self.abi_meta.instrument_meta.y,
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
        interpolation="lanczos",
        gdal_compression_algo="lzw",
        gdal_compression_predictor=2,
    ):
        resampled = self.resample2latlon(
            abi_array, interpolation=interpolation, convert_np=False
        )

        # save COG-compliant GeoTIFF
        # https://www.cogeo.org/providers-guide.html
        # TODO: use new COG driver in GDAL 3.1
        final_gdal_options = self._final_gdal_options + [
            "COPY_SRC_OVERVIEWS=YES",
            "TILED=YES",
            f"COMPRESS={gdal_compression_algo}",
            f"PREDICTOR={gdal_compression_predictor}",
            f"NUM_THREADS={NUM_CPUS}",
        ]
        gdal.SetConfigOption("COMPRESS_OVERVIEW", gdal_compression_algo.upper())
        gdal.SetConfigOption("PREDICTOR_OVERVIEW", str(gdal_compression_predictor))
        resampled.BuildOverviews(interpolation, self._cog_overviews())
        drv = gdal.GetDriverByName("GTiff")
        drv.CreateCopy(str(cog_filepath), resampled, options=final_gdal_options)

        return cog_filepath
