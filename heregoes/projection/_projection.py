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

from os import PathLike
from typing import Annotated, Optional

import numpy as np
from numpy.typing import NDArray
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
    Warp Numpy arrays to and from the projection of an ABI scene.
    Inherits from `heregoes.navigation.ABINavigation`
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
    def y_image_bounds(self) -> NDArray:
        """
        Vertical extents of the ABI scene in radians
        """
        if self._y_image_bounds is None:
            self._set_bounds()
        return self._y_image_bounds

    @property
    def x_image_bounds(self) -> NDArray:
        """
        Horizontal extents of the ABI scene in radians
        """
        if self._x_image_bounds is None:
            self._set_bounds()
        return self._x_image_bounds

    @property
    def y_projected_bounds(self) -> NDArray:
        """
        Vertical extents of the projected ABI scene in false northing and easting (meters)
        """
        if self._y_projected_bounds is None:
            self._set_bounds()
        return self._y_projected_bounds

    @property
    def x_projected_bounds(self) -> NDArray:
        """
        Horizontal extents of the projected ABI scene in false northing and easting (meters)
        """
        if self._x_projected_bounds is None:
            self._set_bounds()
        return self._x_projected_bounds

    @property
    def image_shape_px(self) -> tuple[int, int]:
        """
        Height, width of the ABI scene in Fixed Grid pixels
        """
        if self._image_shape_px is None:
            self._image_shape_px = self.y_rad.size, self.x_rad.size

        return self._image_shape_px

    def resample2cog(
        self,
        source: str | NDArray,
        filepath: PathLike | str,
        resample_algo: str = "lanczos",
        **kwargs,
    ) -> PathLike:
        """
        Resample an `NDArray` from geostationary to equirectangular projection and save to a Cloud-Optimized GeoTIFF (COG)

        #### Parameters:
        - `source`: `NDArray` to resample
        - `filepath`: `str` or `PathLike` object to save the TIFF to
        - `resample_algo` (optional): [GDAL interpolation method](https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-r) to use during the warp
        """
        resampled = self.resample(
            source,
            target="latlon",
            resample_algo=resample_algo,
            return_type="gdal",
            **kwargs,
        )

        return gdal2cog(
            gdal_dataset=resampled,
            filepath=filepath,
            overview_resampling_algo=resample_algo,
        )

    def resample2latlon(
        self, source: str | NDArray, resample_algo: str = "bilinear", **kwargs
    ) -> NDArray:
        """
        Resample an `NDArray` from geostationary to equirectangular projection

        #### Parameters:
        - `source`: `NDArray` to resample
        - `resample_algo` (optional): [GDAL interpolation method](https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-r) to use during the warp
        """
        resampled = self.resample(
            source, target="latlon", resample_algo=resample_algo, **kwargs
        )

        return resampled

    def resample2abi(
        self,
        source: NDArray,
        resample_algo: str = "nearest",
        lat_bounds: tuple[float, float] | Annotated[list[float], 2] = [
            90.0,
            -90.0,
        ],
        lon_bounds: tuple[float, float] | Annotated[list[float], 2] = [
            -180.0,
            180.0,
        ],
        **kwargs,
    ) -> NDArray:
        """
        Resample an `NDArray` from equirectangular to the geostationary projection of this ABI scene

        #### Parameters:
        - `source`: `NDArray` to resample
        - `lat_bounds`, `lon_bounds` (optional): Upper left and lower right lat/lon extents of the equirectangular source data; all coordinates must lie within the ABI scene
            - `lat_bounds=[ul_lat, lr_lat]`, `lon_bounds=[ul_lon, lr_lon]`
        - `resample_algo` (optional): [GDAL interpolation method](https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-r) to use during the warp
        """
        resampled = self.resample(
            source,
            target="abi",
            resample_algo=resample_algo,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            **kwargs,
        )

        return resampled

    def resample(
        self,
        source: str | NDArray,
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
        # https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-r

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

        resample_algos = [
            "near",
            "nearest",
            "bilinear",
            "cubic",
            "cubicspline",
            "lanczos",
            "average",
            "rms",
            "mode",
            "max",
            "min",
            "med",
            "q1",
            "q3",
            "sum",
        ]
        if resample_algo not in resample_algos:
            raise ValueError(
                f"`resample_algo` '{resample_algo}' not supported by GDAL, must be one of:\n{resample_algos}."
            )

        match target.lower():
            case "latlon":
                srcSRS = self._abi_srs
                dstSRS = self._latlon_srs

                # translate options
                translate_outputBounds = [scan_ul_x, scan_ul_y, scan_lr_x, scan_lr_y]

                # warp options
                # 2026: Best to provide explicit bounds but it seems GDAL can't georeference correctly if the bounds contain an off-earth pixel
                warp_outputBounds = [
                    np.nanmin(self.lon_deg),
                    np.nanmax(self.lat_deg),
                    np.nanmax(self.lon_deg),
                    np.nanmin(self.lat_deg),
                ]
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
            resampleAlg=None,
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
            warpMemoryLimit=500,
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
