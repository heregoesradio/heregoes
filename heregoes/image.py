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

"""Basic classes for creating ABI and SUVI imagery"""

import logging
import re
from pathlib import Path

import cv2
import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy import ndimage

from heregoes import exceptions, load, navigation
from heregoes.goesr import abi, suvi
from heregoes.util import align_idx, make_8bit, scale_idx

logger = logging.getLogger("heregoes-logger")
safe_time_format = "%Y-%m-%dT%H%M%SZ"


class Image:
    def __init__(self):
        self.rad = None
        self.dqf = None
        self.quality = None
        self._bv = None

    def save(self, file_path=Path("."), file_ext=".png"):
        file_path = Path(file_path)

        # if a directory path is provided instead of a file path, append a default filename
        if file_path.is_dir():
            file_path = file_path.joinpath(self.default_filename)

        # append an image suffix if not in the file path
        if not (file_path.suffix):
            file_path = file_path.with_suffix(file_ext)

        file_dir = file_path.parent.resolve()
        file_dir.mkdir(parents=True, exist_ok=True)

        if bool(re.search(r"\.jp[e]?g", file_path.suffix.lower())):
            cv2_quality = [cv2.IMWRITE_JPEG_QUALITY, 100]

        elif bool(re.search(r"\.png", file_path.suffix.lower())):
            cv2_quality = [cv2.IMWRITE_PNG_COMPRESSION, 9]

        else:
            cv2_quality = None

        try:
            result = cv2.imwrite(str(file_path), self.bv, cv2_quality)
            if not result:
                raise exceptions.HereGOESIOError
        except Exception as e:
            raise exceptions.HereGOESIOWriteException(
                caller=f"{__name__}.{self.__class__.__name__}",
                filepath=file_path,
                exception=e,
            )


class ABIImage(Image):
    def __init__(
        self,
        abi_nc,
        index=None,
        lat_bounds=None,
        lon_bounds=None,
        gamma=1.0,
        black_space=False,
    ):
        """
        Creates Cloud Moisture Imagery (CMI) following the CMIP ATBD: https://www.star.nesdis.noaa.gov/goesr/docs/ATBD/Imagery.pdf
            - Spectral radiance is stored in `rad`
            - The Data Quality Flag array is stored in `dqf`
            - A ratio of masked to total array elements is stored as `quality`
            - The Cloud Moisture Imagery quantity is stored in `cmi` for:
                - Reflective ABI bands as Reflectance Factor (RF)
                - Emissive ABI bands as Brightness Temperature (BT)
            - "Brightness Value" (BV) is the 8-bit representation of CMI. It is stored in `bv` for:
                - Reflectance factor with a gamma correction
                - Brightness temperature with a bilinear tone curve

        Arguments:
            - `abi_nc`: String or Path object pointing to a GOES-R ABI L1b Radiance netCDF file
            - `index`: Optionally constrains the ABI image to an array index or continuous slice on the ABI Fixed Grid matching the resolution in the provided `abi_nc` file
            - `lat_bounds`, `lon_bounds`: Optionally constrains the ABI image to a latitude and longitude bounding box defined by the upper left and lower right points, e.g. `lat_bounds=[ul_lat, lr_lat]`, `lon_bounds=[ul_lon, lr_lon]`
            - `gamma`: Optional gamma correction for reflective ABI brightness value. Defaults to no correction
            - `black_space`: Optionally overwrites the masked pixels in the final ABI image (nominally the "space" background) to be black. Defaults to no overwriting, or white pixels for reflective imagery and black pixels for emissive imagery. Default `True`
        """

        super().__init__()
        self.index = index
        self.gamma = gamma
        self.black_space = black_space

        self._cmi = None

        self.abi_data = load(abi_nc)

        if lat_bounds is not None and lon_bounds is not None:
            self.abi_nav = navigation.ABINavigation(
                self.abi_data,
                lat_bounds=np.atleast_1d(lat_bounds),
                lon_bounds=np.atleast_1d(lon_bounds),
            )
            self.index = self.abi_nav.index
            self.lat_deg = self.abi_nav.lat_deg
            self.lon_deg = self.abi_nav.lon_deg

        elif self.index is None:
            self.index = np.s_[:, :]

        self.rad = self.abi_data["Rad"][self.index]
        self.dqf = self.abi_data["DQF"][self.index]
        self.mask = self.abi_data["Rad"].mask
        self.quality = self.abi_data["Rad"].pct_unmasked

        self.rad_range = np.array(
            self.abi_data["Rad"].valid_range * self.abi_data["Rad"].scale_factor
            + self.abi_data["Rad"].add_offset,
            dtype=np.float32,
        )

        self.default_filename = "_".join(
            (
                self.abi_data.platform_ID.lower(),
                self.abi_data.instrument_type_safe.lower(),
                self.abi_data.scene_id_safe.lower(),
                self.abi_data.band_id_safe.lower(),
                self.abi_data.time_coverage_start.strftime(safe_time_format),
            )
        )

    @property
    def cmi(self):
        if self._cmi is None:
            if 1 <= self.abi_data["band_id"][...] <= 6:
                self._cmi = abi.rad2rf(
                    self.rad,
                    self.abi_data["earth_sun_distance_anomaly_in_AU"][...].item(),
                    self.abi_data["esun"][...].item(),
                )

            elif 7 <= self.abi_data["band_id"][...] <= 16:
                self._cmi = abi.rad2bt(
                    self.rad,
                    self.abi_data["planck_fk1"][...].item(),
                    self.abi_data["planck_fk2"][...].item(),
                    self.abi_data["planck_bc1"][...].item(),
                    self.abi_data["planck_bc2"][...].item(),
                )

        return self._cmi

    @cmi.setter
    def cmi(self, value):
        self._cmi = value

    @property
    def bv(self):
        if self._bv is None:
            if 1 <= self.abi_data["band_id"][...] <= 6:
                # calculate the range of possible reflectance factors from the provided valid range of radiance, and use it to normalize before the gamma correction
                self.rf_min, self.rf_max = (
                    self.rad_range
                    * np.pi
                    * np.square(self.abi_data["earth_sun_distance_anomaly_in_AU"][...])
                ) / self.abi_data["esun"][...]
                self._bv = abi.rf2bv(
                    self.cmi, min=self.rf_min, max=self.rf_max, gamma=self.gamma
                )

            elif 7 <= self.abi_data["band_id"][...] <= 16:
                self._bv = abi.bt2bv(self.cmi)

        if self.black_space:
            self._bv[self.mask] = 0

        return self._bv

    @bv.setter
    def bv(self, value):
        self._bv = value


class ABINaturalRGB(Image):
    def __init__(
        self,
        red_nc,
        green_nc,
        blue_nc,
        index=None,
        lat_bounds=None,
        lon_bounds=None,
        r_coeff=0.45,
        g_coeff=0.1,
        b_coeff=0.45,
        upscale=False,
        upscale_algo=cv2.INTER_CUBIC,
        gamma=1.0,
        black_space=False,
    ):
        """
        Creates the "natural" color RGB for ABI following https://doi.org/10.1029/2018EA000379 in BGR order

        Arguments:
            - `red_nc`, `green_nc`, `blue_nc`: Strings or Path objects pointing to GOES-R ABI L1b Radiance netCDF files for the red (0.64 μm), green (0.86 μm), and blue (0.47 μm) components
            - `index`: Optionally constrains the ABI imagery to an array index or continuous slice on the ABI Fixed Grid. If `upscale` is `False` (default), this is an index on the 1 km Fixed Grid. Otherwise, if `upscale` is `True`, a 500 m Fixed Grid is used
            - `lat_bounds`, `lon_bounds`: Optionally constrains the ABI imagery to a latitude and longitude bounding box defined by the upper left and lower right points, e.g. `lat_bounds=[ul_lat, lr_lat]`, `lon_bounds=[ul_lon, lr_lon]`
            - `r_coeff`, `g_coeff`, `b_coeff`: Coefficients for the fractional combination "green" band method described in Bah et. al (2018)
            - `upscale`: Whether to scale up green and blue images (1 km) to match the red image (500 m) (`True`) or vice versa (`False`, Default)
            - `upscale_algo`: The OpenCV interpolation algorithm used for upscaling green and blue images. See https://docs.opencv.org/4.6.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121. Default `cv2.INTER_CUBIC`
            - `gamma`: Optional gamma correction for reflective ABI brightness value. Defaults to no correction
            - `black_space`: Optionally overwrites the masked pixels in the final ABI image (nominally the "space" background) to be black. Defaults to no overwriting, or white pixels for reflective imagery and black pixels for emissive imagery. Default `True`
        """

        super().__init__()
        self.index = index

        if upscale:
            nav_nc = red_nc
            scaler_500m = 1.0
            scaler_1km = 2.0

        else:
            nav_nc = green_nc
            scaler_500m = 0.5
            scaler_1km = 1.0

        if lat_bounds is not None and lon_bounds is not None:
            self.abi_nav = navigation.ABINavigation(
                load(nav_nc),
                lat_bounds=np.atleast_1d(lat_bounds),
                lon_bounds=np.atleast_1d(lon_bounds),
            )
            self.index = self.abi_nav.index
            self.lat_deg = self.abi_nav.lat_deg
            self.lon_deg = self.abi_nav.lon_deg

        elif self.index is None:
            self.index = np.s_[:, :]

        if upscale:
            aligned_idx = align_idx(self.index, 2)
            if self.index != aligned_idx:
                # only warn for index alignment if index was provided
                if lat_bounds is None or lon_bounds is None:
                    logger.warning(
                        "Adjusting provided index %s to %s to align to the 1 km ABI Fixed Grid.",
                        str(self.index),
                        str(aligned_idx),
                        extra={"caller": f"{__name__}.{self.__class__.__name__}"},
                    )
                self.index = aligned_idx

        # make images
        red_image = ABIImage(
            red_nc,
            index=scale_idx(self.index, 1 / scaler_500m),
            gamma=gamma,
            black_space=black_space,
        )
        green_image = ABIImage(
            green_nc,
            index=scale_idx(self.index, 1 / scaler_1km),
            gamma=gamma,
            black_space=black_space,
        )
        blue_image = ABIImage(
            blue_nc,
            index=scale_idx(self.index, 1 / scaler_1km),
            gamma=gamma,
            black_space=black_space,
        )

        # resize red
        red_image.bv = cv2.resize(
            red_image.bv,
            None,
            fx=scaler_500m,
            fy=scaler_500m,
            interpolation=cv2.INTER_AREA,
        )
        red_image.dqf = cv2.resize(
            red_image.dqf,
            None,
            fx=scaler_500m,
            fy=scaler_500m,
            interpolation=cv2.INTER_NEAREST,
        )
        red_image.mask = (
            cv2.resize(
                red_image.mask.astype(np.uint8),
                None,
                fx=scaler_500m,
                fy=scaler_500m,
                interpolation=cv2.INTER_NEAREST,
            )
            == 1
        )

        # resize green
        green_image.bv = cv2.resize(
            green_image.bv,
            None,
            fx=scaler_1km,
            fy=scaler_1km,
            interpolation=upscale_algo,
        )
        green_image.dqf = cv2.resize(
            green_image.dqf,
            None,
            fx=scaler_1km,
            fy=scaler_1km,
            interpolation=cv2.INTER_NEAREST,
        )
        green_image.mask = (
            cv2.resize(
                green_image.mask.astype(np.uint8),
                None,
                fx=scaler_1km,
                fy=scaler_1km,
                interpolation=cv2.INTER_NEAREST,
            )
            == 1
        )

        # resize blue
        blue_image.bv = cv2.resize(
            blue_image.bv,
            None,
            fx=scaler_1km,
            fy=scaler_1km,
            interpolation=upscale_algo,
        )
        blue_image.dqf = cv2.resize(
            blue_image.dqf,
            None,
            fx=scaler_1km,
            fy=scaler_1km,
            interpolation=cv2.INTER_NEAREST,
        )
        blue_image.mask = (
            cv2.resize(
                blue_image.mask.astype(np.uint8),
                None,
                fx=scaler_1km,
                fy=scaler_1km,
                interpolation=cv2.INTER_NEAREST,
            )
            == 1
        )

        green_image.bv = (
            (red_image.bv * r_coeff)
            + (green_image.bv * g_coeff)
            + (blue_image.bv * b_coeff)
        )

        self.bv = make_8bit(
            np.stack([blue_image.bv, green_image.bv, red_image.bv], axis=2)
        )
        self.quality = (
            sum([red_image.quality, green_image.quality, blue_image.quality]) / 3
        )
        self.dqf = np.stack([blue_image.dqf, green_image.dqf, red_image.dqf], axis=2)
        self.mask = np.stack(
            [blue_image.mask, green_image.mask, red_image.mask], axis=2
        )

        if upscale:
            self.abi_data = red_image.abi_data

        else:
            self.abi_data = green_image.abi_data

        self.abi_data["band_id"][...] = np.atleast_1d(0)
        self.abi_data.band_id_safe = "Color"

        self.abi_data.dataset_name = "RGB from " + ", ".join(
            (str(red_nc), str(green_nc), str(blue_nc))
        )
        self.abi_data.rgb = [str(red_nc), str(green_nc), str(blue_nc)]

        self.default_filename = "_".join(
            (
                self.abi_data.platform_ID.lower(),
                self.abi_data.instrument_type_safe.lower(),
                self.abi_data.scene_id_safe.lower(),
                self.abi_data.band_id_safe.lower(),
                self.abi_data.time_coverage_start.strftime(safe_time_format),
            )
        )


class SUVIImage(Image):
    def __init__(
        self,
        suvi_nc,
        shift=True,
        shift_limit=100,
        flip=True,
        dqf_correction=True,
    ):
        """
        Creates a 1-second 8-bit SUVI image made to look similar to what is shown on the SWPC website: https://www.swpc.noaa.gov/products/goes-solar-ultraviolet-imager-suvi

        Arguments:
            - `suvi_nc`: String or Path object pointing to a 1-second exposure GOES-R SUVI L1b Solar Imagery netCDF file
            - `shift`: Whether to try moving the center of the Sun to the center of the image. Default `True`
            - `shift_limit` Limits the shift operation that moves the Sun to the center of the image to a maximum of `shift_limit` pixels. Default `100`
            - `flip`: Whether to flip the SUVI image from S-N to N-S to match SWPC. Default `True`
            - `dqf_correction`: Whether to interpolate over bad pixels marked by DQF. Default `True`
        """

        super().__init__()

        self.suvi_data = load(suvi_nc)

        if self.suvi_data["CMD_EXP"][...] != 1.0:
            logger.warning(
                "Short SUVI exposure detected: SUVI exposures shorter than 1 second are not officially supported.",
                extra={"caller": f"{__name__}.{self.__class__.__name__}"},
            )

        self.rad = self.suvi_data["RAD"][...]
        self.dqf = self.suvi_data["DQF"][...]
        self.quality = self.suvi_data["RAD"].pct_unmasked
        x_offset = 640 - self.suvi_data["CRPIX1"][...]
        y_offset = 640 - self.suvi_data["CRPIX2"][...]

        self.default_filename = "_".join(
            (
                self.suvi_data.platform_ID.lower(),
                self.suvi_data.instrument_type_safe.lower(),
                self.suvi_data.wavelength_safe.lower(),
                self.suvi_data.time_coverage_start.strftime(safe_time_format),
            )
        )

        if dqf_correction:
            # first we dilate the DQF so it covers the bad pixel halos (3x3 kernel)
            dilated_dqf = ndimage.binary_dilation(
                self.dqf, structure=np.ones((3, 3), np.uint8), iterations=1
            )
            # replace flagged pixels with NaN, then interpolate over NaN with 9x9 Gaussian kernel
            self.rad = interpolate_replace_nans(
                np.where(dilated_dqf != 0, np.nan, self.rad),
                Gaussian2DKernel(x_stddev=1, x_size=9, y_size=9),
            )

        if shift:
            # move the sun to the center of the image using default ndimage spline interpolation to a maximum of `shift_limit` pixels
            self.rad = ndimage.shift(
                self.rad,
                (
                    np.sign(y_offset) * min(np.abs(y_offset), shift_limit),
                    np.sign(x_offset) * min(np.abs(x_offset), shift_limit),
                ),
                mode="constant",
                cval=0.0,
            )

        if flip:
            # SUVI arrays are S-N, make them N-S to match SWPC
            self.rad = np.flipud(self.rad)

    @property
    def bv(self):
        if self._bv is None:
            self._bv = suvi.rad2bv(
                self.rad,
                *self.suvi_data.instrument_coefficients.input_range,
                self.suvi_data.instrument_coefficients.asinh_a,
                *self.suvi_data.instrument_coefficients.output_range,
            )

        return self._bv

    @bv.setter
    def bv(self, value):
        self._bv = value
