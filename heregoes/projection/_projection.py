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

from typing import Annotated, Optional

import numpy as np
import numpy.typing as npt
from osgeo import gdal

gdal.UseExceptions()

from heregoes.core import NUM_CPUS, PARALLEL_MODE
from heregoes.core.types import ABIInputType, FixedGridDataType, FixedGridIndexType
from heregoes.navigation import ABINavigation
from heregoes.projection._funcs import (
    gdal2cog,
    gdal2numpy,
    numpy2gdal,
    translate_and_warp,
)


class ABIProjection(ABINavigation):
    """
    ### Resample Numpy arrays to and from the projection of an ABI scene

    ### Parameters:
        - `abi_data`:
            - Either a str or Path referencing an ABI L1b/L2+ netCDF file,
            - or the `ABIL1bData` or `ABIL2Data` object formed by `heregoes.load()` on the path

        - `index` (optional): 2D array index or slice to select a subset of the ABI Fixed Grid, e.g.:
            - `np.s_[y1:y2, x1:x2]`
            - `(slice(y1, y2, None), slice(x1, x2, None))`

        - `lat_bounds`, `lon_bounds` (optional): Instead of `index`, use geodetic latitude and longitude (degrees) to select a slice of the Fixed Grid, e.g.:
            - `lat_bounds=[ul_lat, lr_lat]`, `lon_bounds=[ul_lon, lr_lon]`

        - `height_m` (optional): If subsetting the projection by `index` or `lat_bounds` and `lon_bounds`, provide the height in meters relative to the GRS80 at the bounding points. Default 0.0 (no correction)
            - `height_m=[ul_m, lr_m]`

    ### Attributes:
        - `y_image_bounds`, `x_image_bounds`:
            - The vertical and horizontal extents of the ABI image in radians
        - `y_projected_bounds`, `x_projected_bounds`:
            - The vertical and horizontal extents of the ABI projection in false northing and easting (meters)
    """

    def __init__(
        self,
        abi_data: ABIInputType,
        index: Optional[FixedGridIndexType] = None,
        lat_bounds: Optional[FixedGridDataType] = None,
        lon_bounds: Optional[FixedGridDataType] = None,
        height_m: FixedGridDataType = 0.0,
        **kwargs,
    ):
        super().__init__(
            abi_data,
            index=index,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            height_m=height_m,
            **kwargs,
        )

        self._y_image_bounds = None
        self._x_image_bounds = None
        self._y_projected_bounds = None
        self._x_projected_bounds = None
        self._image_shape_px = None

        h = self.abi_data["goes_imager_projection"].perspective_point_height
        a = self.abi_data["goes_imager_projection"].semi_major_axis
        b = self.abi_data["goes_imager_projection"].semi_minor_axis
        f = 1 / self.abi_data["goes_imager_projection"].inverse_flattening
        lat_0 = self.abi_data["goes_imager_projection"].latitude_of_projection_origin
        lon_0 = self.abi_data["goes_imager_projection"].longitude_of_projection_origin
        sweep = self.abi_data["goes_imager_projection"].sweep_angle_axis

        self._latlon_srs = "+proj=latlon +ellps=WGS84 +datum=WGS84 +no_defs"
        self._abi_srs = f"+proj=geos +h={h} +a={a} +b={b} +f={f} +lat_0={lat_0} +lon_0={lon_0} +x_0=0.0 y_0=0.0 +sweep={sweep} +ellps=GRS80 +no_defs"

    def _set_bounds(self):
        # set the projection bounds for this image given that it may be subsetted

        # the full scanning angle bounds are given by offsetting with half of the pixel IFOV on all sides
        offset = self.abi_data.resolution_ifov / np.float32(2)
        self._y_image_bounds = np.array(
            [self.y_rad[0] + offset, self.y_rad[-1] - offset], dtype=np.float64
        )
        self._x_image_bounds = np.array(
            [self.x_rad[0] - offset, self.x_rad[-1] + offset], dtype=np.float64
        )

        # multiply by the satellite height to get the projected bounds as false northing, easting (meters)
        h = self.abi_data["goes_imager_projection"].perspective_point_height
        self._y_projected_bounds = self._y_image_bounds * h
        self._x_projected_bounds = self._x_image_bounds * h

    @property
    def y_image_bounds(self):
        if self._y_image_bounds is None:
            self._set_bounds()
        return self._y_image_bounds

    @property
    def x_image_bounds(self):
        if self._x_image_bounds is None:
            self._set_bounds()
        return self._x_image_bounds

    @property
    def y_projected_bounds(self):
        if self._y_projected_bounds is None:
            self._set_bounds()
        return self._y_projected_bounds

    @property
    def x_projected_bounds(self):
        if self._x_projected_bounds is None:
            self._set_bounds()
        return self._x_projected_bounds

    @property
    def image_shape_px(self):
        if self._image_shape_px is None:
            self._image_shape_px = self.y_rad.size, self.x_rad.size

        return self._image_shape_px

    def resample2cog(self, source, filepath, resample_algo="lanczos", **kwargs):
        resampled = self.resample2latlon(
            source,
            resample_algo=resample_algo,
            return_type="gdal",
            **kwargs,
        )

        return gdal2cog(
            gdal_dataset=resampled,
            filepath=filepath,
            overview_resampling_algo=resample_algo,
        )

    def resample2latlon(self, source, resample_algo="bilinear", **kwargs):
        resampled = self.resample(
            source, target="latlon", resample_algo=resample_algo, **kwargs
        )

        return resampled

    def resample2abi(self, source, resample_algo="nearest", **kwargs):
        resampled = self.resample(
            source, target="abi", resample_algo=resample_algo, **kwargs
        )

        return resampled

    def resample(
        self,
        source: str | npt.NDArray,
        target: str,
        resample_algo: str = "bilinear",
        return_type: str = "numpy",
        lat_bounds: tuple[float, float] | Annotated[list[float], 2] = [
            90.0,
            -90.0,
        ],
        lon_bounds: tuple[float, float] | Annotated[list[float], 2] = [
            -180.0,
            180.0,
        ],
        source_nodata: Optional[float] = None,
        target_nodata: Optional[float] = None,
    ):

        intermediate_format = "GTiff"
        intermediate_gdal_options = ["COMPRESS=NONE"]

        if isinstance(source, str):
            source = getattr(self, source)

        elif source.ndim == 3 and source.shape[-1] == 3:
            intermediate_gdal_options += ["PHOTOMETRIC=RGB"]

        if 1 in self.image_shape_px:
            raise ValueError(
                f"Image with shape {self.image_shape_px} is not projectable."
            )

        image_height_px, image_width_px = self.image_shape_px
        scan_ul_y, scan_lr_y = self.y_projected_bounds
        scan_ul_x, scan_lr_x = self.x_projected_bounds

        match target.lower():
            case "latlon":
                srcSRS = self._abi_srs
                dstSRS = self._latlon_srs

                # translate options
                translate_outputBounds = [scan_ul_x, scan_ul_y, scan_lr_x, scan_lr_y]

                # warp options
                warp_outputBounds = None
                width = 0
                height = 0

                # Set the projection resolution to that of ABI at the equator for consistency
                # This seems crude, but without it GDAL estimates the output resolution based on the output bounds which change between ABI scenes or subsets thereof
                c_eq = (
                    2.0
                    * np.pi
                    * self.abi_data.variables.goes_imager_projection.semi_major_axis
                )
                meters_per_degree = c_eq / 360.0
                nadir_resolution_meters = (
                    self.abi_data.resolution_km * 1000.0
                )  # horizontal
                degrees_per_pixel = nadir_resolution_meters / meters_per_degree
                xRes = degrees_per_pixel
                yRes = degrees_per_pixel

            case "abi":
                srcSRS = self._latlon_srs
                dstSRS = self._abi_srs

                ul_lat, lr_lat = lat_bounds
                ul_lon, lr_lon = lon_bounds

                # translate options
                translate_outputBounds = [ul_lon, ul_lat, lr_lon, lr_lat]

                # warp options
                warp_outputBounds = [scan_ul_x, scan_ul_y, scan_lr_x, scan_lr_y]
                width = image_width_px
                height = image_height_px
                xRes = None
                yRes = None

            case _:
                raise ValueError("`target` must be one of 'abi', 'latlon'.")

        translate_options = gdal.TranslateOptions(
            outputSRS=srcSRS,
            outputBounds=translate_outputBounds,
            format=intermediate_format,
            resampleAlg=resample_algo.lower(),
            creationOptions=intermediate_gdal_options,
            noData=source_nodata,
        )
        warp_options = gdal.WarpOptions(
            srcSRS=srcSRS,
            dstSRS=dstSRS,
            outputBounds=warp_outputBounds,
            width=width,
            height=height,
            format=intermediate_format,
            resampleAlg=resample_algo.lower(),
            creationOptions=intermediate_gdal_options,
            multithread=PARALLEL_MODE,
            warpOptions=[f"NUM_THREADS={NUM_CPUS}"],
            errorThreshold=0,
            xRes=xRes,
            yRes=yRes,
            srcNodata=source_nodata,
            dstNodata=target_nodata,
        )

        dataset = numpy2gdal(source)
        dataset.SetProjection(srcSRS)
        resampled = translate_and_warp(
            gdal_dataset=dataset,
            translate_options=translate_options,
            warp_options=warp_options,
        )

        match return_type.lower():
            case "numpy":
                resampled = gdal2numpy(resampled)
                if target == "abi":
                    resampled = np.flipud(resampled)
                return resampled

            case "gdal":
                return resampled

            case _:
                raise ValueError("`return_type` must be one of 'numpy', 'gdal'.")
