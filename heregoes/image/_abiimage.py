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

import logging
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from heregoes.core.types import ABIL1bInputType, FixedGridDataType, FixedGridIndexType
from heregoes.goesr import abi
from heregoes.image._image import _Image
from heregoes.projection import ABIProjection
from heregoes.util import align_idx, linear_norm, scale_arr, scale_idx

logger = logging.getLogger()
safe_time_format = "%Y-%m-%dT%H%M%SZ"


class _BaseABIImage(_Image, ABIProjection):
    def __init__(
        self,
        abi_data: ABIL1bInputType,
        index: Optional[FixedGridIndexType] = None,
        lat_bounds: Optional[FixedGridDataType] = None,
        lon_bounds: Optional[FixedGridDataType] = None,
        height_m: Optional[FixedGridDataType] = 0.0,
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

    def resample2cog(self, filepath, **kwargs):
        return super().resample2cog(
            source="bv",
            filepath=filepath,
            **kwargs,
        )

    def resample2latlon(self, **kwargs):
        return super().resample2latlon(
            source="bv",
            **kwargs,
        )


class ABIImage(_BaseABIImage):
    """
    Creates 8-bit Cloud and Moisture Imagery (CMI) following the [ATBD](https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Enterprise/ATBD_Enterprise_Cloud_and_Moisture_Imagery_Product_v4_2021-01-13.pdf).
    Inherits from `heregoes.projection.ABIProjection` and `heregoes.navigation.ABINavigation`
    """

    def __init__(
        self,
        abi_data: ABIL1bInputType,
        index: Optional[FixedGridIndexType] = None,
        lat_bounds: Optional[FixedGridDataType] = None,
        lon_bounds: Optional[FixedGridDataType] = None,
        height_m: Optional[FixedGridDataType] = 0.0,
        gamma: float = 1.0,
        normalize_rf: bool = False,
        black_space: bool = False,
        **kwargs,
    ):
        """
        #### Parameters:
        - `abi_data`: Either a `str` or `Path` referencing an ABI L1b netCDF file, or one already loaded by `heregoes.load()`

        - `index` (optional): 2D slice to select a subset of the ABI image, e.g.:
            - `np.s_[y1:y2, x1:x2]`

        - `lat_bounds`, `lon_bounds` (optional): Instead of `index`, use a geographic bounding box to select a slice of the ABI image, e.g.:
            - `lat_bounds=[ul_lat, lr_lat]`, `lon_bounds=[ul_lon, lr_lon]`

        - `height_m` (optional): If subsetting the image by `index` or `lat_bounds` and `lon_bounds`, provide the height in meters relative to the GRS80 at the bounding points. Default 0.0 (no parallax correction)
            - `height_m=1234.0`
            - `height_m=[ul_m, lr_m]`

        - `gamma` (optional): Gamma correction term; 0.5 is the common square root enhancement. Default `1.0` or no correction

        - `normalize_rf` (optional): Whether to linearly normalize reflectance factor (RF) in the output image to preserve highlights. Default `False`, but RF is always normalized when `gamma` is 1.0

        - `black_space` (optional): Whether to overwrite masked pixels in the final ABI image (nominally the "space" background) to be black. Defaults to no overwriting, or white pixels for reflective imagery and black pixels for emissive imagery. Default `False`
        """
        super().__init__(
            abi_data,
            index=index,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            height_m=height_m,
            **kwargs,
        )
        self.gamma = gamma
        self.normalize_rf = normalize_rf
        self.black_space = black_space

        self._rad = None
        self._dqf = None
        self._cmi = None
        self._bv = None

        self._rad = self.abi_data["Rad"][self.index]
        self.mask: NDArray = self.abi_data["Rad"].mask

        self.quality: float = self.abi_data["Rad"].pct_unmasked
        """
        The ratio of masked to total image array elements
        """

        rad_range = np.array(
            self.abi_data["Rad"].valid_range * self.abi_data["Rad"].scale_factor
            + self.abi_data["Rad"].add_offset,
            dtype=np.float32,
        )

        self._rf_min, self._rf_max = (
            rad_range
            * np.pi
            * np.square(self.abi_data["earth_sun_distance_anomaly_in_AU"][...])
        ) / self.abi_data["esun"][...]

        self.default_filename = "_".join(
            (
                self.abi_data.platform_ID.lower(),
                self.abi_data._instrument_type_str.lower(),
                self.abi_data._scene_id_str.lower(),
                self.abi_data._band_id_str.lower(),
                self.abi_data.time_coverage_start.strftime(safe_time_format),
            )
        )

    @property
    def rad(self) -> NDArray:
        """
        Spectral radiance in either `W m-2 sr-1 um-1` for ABI bands 1-6,
        or `mW m-2 sr-1 (cm-1)-1` for emissive bands 7-16
        """
        if self._rad is None:
            self._rad = self.abi_data["Rad"][self.index]

        return self._rad

    @rad.setter
    def rad(self, value):
        self._rad = value

    @property
    def dqf(self) -> NDArray:
        """
        Data Quality Flag array delivered with the product
        """
        if self._dqf is None:
            self._dqf = self.abi_data["DQF"][self.index]

        return self._dqf

    @dqf.setter
    def dqf(self, value):
        self._dqf = value

    @property
    def cmi(self) -> NDArray:
        """
        Cloud and Moisture Imagery; either the Lambertian-equivalent reflectance factor (RF) for ABI bands 1-6,
        or blackbody brightness temperature (BT) in Kelvin for emissive bands 7-16
        """
        if self._cmi is None:
            if 1 <= self.abi_data["band_id"][...] <= 6:
                self._cmi = abi.rad2rf(
                    self.rad,
                    self.abi_data["earth_sun_distance_anomaly_in_AU"][...],
                    self.abi_data["esun"][...],
                )

            elif 7 <= self.abi_data["band_id"][...] <= 16:
                self._cmi = abi.rad2bt(
                    self.rad,
                    self.abi_data["planck_fk1"][...],
                    self.abi_data["planck_fk2"][...],
                    self.abi_data["planck_bc1"][...],
                    self.abi_data["planck_bc2"][...],
                )

        self.rad = None

        return self._cmi

    @cmi.setter
    def cmi(self, value):
        self._cmi = value

    @property
    def bv(self) -> NDArray:
        """
        Pixel nrightness values 0-255 for either reflectance factor with a gamma correction,
        or brightness temperature with a bilinear tone curve
        """
        if self._bv is None:
            if 1 <= self.abi_data["band_id"][...] <= 6:
                if self.normalize_rf or self.gamma == 1.0:
                    self._bv = abi.rf2bv(
                        linear_norm(
                            self.cmi,
                            old_min=0.0,
                            old_max=self._rf_max,
                            new_min=0.0,
                            new_max=1.0,
                        ),
                        gamma=self.gamma,
                    )

                else:
                    self._bv = abi.rf2bv(self.cmi, gamma=self.gamma)

            elif 7 <= self.abi_data["band_id"][...] <= 16:
                self._bv = abi.bt2bv(self.cmi)

            if self.black_space:
                self._bv[self.mask] = 0

        self.cmi = None

        return self._bv

    @bv.setter
    def bv(self, value):
        self._bv = value


class ABINaturalRGB(_BaseABIImage):
    """
    Creates the "natural" color RGB for ABI following [Bah et. al (2018)](https://doi.org/10.1029/2018EA000379)
    Inherits from `heregoes.projection.ABIProjection` and `heregoes.navigation.ABINavigation`
    """

    def __init__(
        self,
        red_data: ABIL1bInputType,
        green_data: ABIL1bInputType,
        blue_data: ABIL1bInputType,
        index: Optional[FixedGridIndexType] = None,
        lat_bounds: Optional[FixedGridDataType] = None,
        lon_bounds: Optional[FixedGridDataType] = None,
        height_m: Optional[FixedGridDataType] = 0.0,
        gamma: float = 1.0,
        normalize_rf: bool = False,
        black_space: bool = False,
        r_coeff: float = 0.45,
        g_coeff: float = 0.1,
        b_coeff: float = 0.45,
        upscale: bool = False,
        upscale_algo: str = "cubic",
        **kwargs,
    ):
        """
        #### Parameters:
        - `red_data`, `green_data`, `blue_data`:
            - For each of the red (0.64 μm), green (0.86 μm), and blue (0.47 μm) radiance components:
                - Either a `str` or `Path` referencing an ABI L1b netCDF file, or one already loaded by `heregoes.load()`

        - `index` (optional): 2D slice to select a subset of the ABI image, e.g.:
            - `np.s_[y1:y2, x1:x2]`

        - `lat_bounds`, `lon_bounds` (optional): Instead of `index`, use a geographic bounding box to select a slice of the ABI image, e.g.:
            - `lat_bounds=[ul_lat, lr_lat]`, `lon_bounds=[ul_lon, lr_lon]`

        - `height_m` (optional): If subsetting the image by `index` or `lat_bounds` and `lon_bounds`, provide the height in meters relative to the GRS80 at the bounding points. Default 0.0 (no parallax correction)
            - `height_m=[ul_m, lr_m]`

        - `gamma` (optional): Gamma correction term; 0.5 is the common square root enhancement. Default `1.0` or no correction

        - `normalize_rf` (optional): Whether to linearly normalize reflectance factor (RF) in the output image to preserve highlights. Default `False`, but RF is always normalized when `gamma` is 1.0

        - `black_space` (optional): Whether to overwrite masked pixels in the final ABI image (nominally the "space" background) to be black. Default `False`

        - `r_coeff`, `g_coeff`, `b_coeff` (optional): Coefficients for the fractional combination "green" band method described in Bah et. al (2018)

        - `upscale` (optional): Whether to scale up green and blue images (1 km) to match the red image (500 m) (`True`) or vice versa (`False`, Default)

        - `upscale_algo` (optional): The OpenCV interpolation algorithm used for upscaling green and blue images, one of "area", "cubic", "lanczos", "linear", "nearest". Default "cubic"
        """
        if upscale:
            nav_data = red_data
            scaler_500m = 1.0
            scaler_1km = 2.0

        else:
            nav_data = green_data
            scaler_500m = 0.5
            scaler_1km = 1.0

        cv2_interpolation = None
        match upscale_algo.lower():
            case "area":
                cv2_interpolation = cv2.INTER_AREA
            case "cubic":
                cv2_interpolation = cv2.INTER_CUBIC
            case "lanczos":
                cv2_interpolation = cv2.INTER_LANCZOS4
            case "linear":
                cv2_interpolation = cv2.INTER_LINEAR
            case "nearest":
                cv2_interpolation = cv2.INTER_NEAREST
            case _:
                raise ValueError(
                    f"Unsupported CV2 interpolation algorithm: {upscale_algo}."
                )

        super().__init__(
            nav_data,
            index=index,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            height_m=height_m,
            **kwargs,
        )

        if upscale:
            aligned_idx = align_idx(self.index, 2)
            if not np.array_equal(self.index, aligned_idx):
                # only warn for index alignment if index was provided
                if self._index_mode:
                    logger.warning(
                        "Adjusting provided index %s to %s to align to the 1 km ABI Fixed Grid.",
                        str(self.index),
                        str(aligned_idx),
                    )
                self.index = aligned_idx

        # make intermediate images
        red_image = ABIImage(
            red_data,
            index=scale_idx(self.index, 1 / scaler_500m),
            gamma=gamma,
            normalize_rf=normalize_rf,
            black_space=black_space,
        )
        red_image.bv = scale_arr(
            red_image.bv,
            k=scaler_500m,
            interpolation=cv2.INTER_AREA,
        )

        green_image = ABIImage(
            green_data,
            index=scale_idx(self.index, 1 / scaler_1km),
            gamma=gamma,
            normalize_rf=normalize_rf,
            black_space=black_space,
        )
        green_image.bv = scale_arr(
            green_image.bv,
            k=scaler_1km,
            interpolation=cv2_interpolation,
        )

        blue_image = ABIImage(
            blue_data,
            index=scale_idx(self.index, 1 / scaler_1km),
            gamma=gamma,
            normalize_rf=normalize_rf,
            black_space=black_space,
        )
        blue_image.bv = scale_arr(
            blue_image.bv,
            k=scaler_1km,
            interpolation=cv2_interpolation,
        )

        self.bv: NDArray = abi.bv2rgb(
            r_bv=red_image.bv,
            g_bv=green_image.bv,
            b_bv=blue_image.bv,
            r_coeff=r_coeff,
            g_coeff=g_coeff,
            b_coeff=b_coeff,
        )
        """
        Color pixel brightness values 0-255 with dimensions in BGR order
        """

        self.quality: float = (
            sum([red_image.quality, green_image.quality, blue_image.quality]) / 3
        )
        """
        The ratio of masked to total image pixels across red, green, and blue components
        """

        # update some meta to reflect that this is an RGB
        self.abi_data["band_id"][...] = np.atleast_1d(0)
        self.abi_data._band_id_str = "Color"

        self.abi_data.rgb = [str(red_data), str(green_data), str(blue_data)]
        self.abi_data.dataset_name = "RGB from " + ", ".join(self.abi_data.rgb)

        self.default_filename = "_".join(
            (
                self.abi_data.platform_ID.lower(),
                self.abi_data._instrument_type_str.lower(),
                self.abi_data._scene_id_str.lower(),
                self.abi_data._band_id_str.lower(),
                self.abi_data.time_coverage_start.strftime(safe_time_format),
            )
        )
