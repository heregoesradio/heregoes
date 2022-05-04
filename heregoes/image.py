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

"""Basic classes for creating ABI and SUVI imagery"""

import re
from pathlib import Path

import cv2
import netCDF4
import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy import ndimage

from heregoes import (
    abi,
    logger,
    meta,
    suvi,
    util,
)


class Image:
    def __init__(self):
        self.rad = None
        self.dqf = None
        self.quality = None
        self._bv = None

    def save(self, file_path):
        file_path = Path(file_path)
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
                raise ValueError(f"Could not save image to {file_path}")

        except Exception as e:
            logger.critical(e)

            return


class ABIImage(Image):
    def __init__(
        self,
        abi_nc,
        index=slice(None, None),
        gamma=1.0,
        mask_fill=False,
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
            - `index`: Optionally process an ABI image for a single array index or slice
            - `gamma`: Optional gamma correction for reflective ABI brightness value. Defaults to no correction
            - `mask_fill` Optionally fills masked radiance with np.nan and masked DQF with 0. Default `False`
        """

        super(ABIImage, self).__init__()
        self.gamma = gamma
        self._cmi = None

        with netCDF4.Dataset(abi_nc, "r") as loaded_abi_nc:
            self.rad = np.atleast_1d(loaded_abi_nc["Rad"][index])
            valid_range = loaded_abi_nc["Rad"].valid_range
            scale_factor = loaded_abi_nc["Rad"].scale_factor
            add_offset = loaded_abi_nc["Rad"].add_offset
            self.dqf = np.atleast_1d(loaded_abi_nc["DQF"][index])

        self.rad_range = np.array(
            valid_range * scale_factor + add_offset, dtype=np.float32
        )

        if self.rad.size > 1:
            # count of non-masked elements over total number of elements
            self.quality = self.rad.count() / self.rad.size

            if mask_fill:
                self.rad[self.rad.mask] = np.nan
                self.dqf[self.dqf.mask] = 0

        self.meta = meta.NCMeta(abi_nc)

    @property
    def cmi(self):
        if self._cmi is None:
            if 1 <= self.meta.instrument_meta.band_id <= 6:
                self._cmi = abi.rad2rf(
                    self.rad,
                    self.meta.instrument_meta.esd,
                    self.meta.instrument_meta.esun,
                )

            elif 7 <= self.meta.instrument_meta.band_id <= 16:
                self._cmi = abi.rad2bt(
                    self.rad,
                    self.meta.instrument_meta.planck_fk1,
                    self.meta.instrument_meta.planck_fk2,
                    self.meta.instrument_meta.planck_bc1,
                    self.meta.instrument_meta.planck_bc2,
                )

        return self._cmi

    @cmi.setter
    def cmi(self, value):
        self._cmi = value

    @property
    def bv(self):
        if self._bv is None:
            if 1 <= self.meta.instrument_meta.band_id <= 6:
                # calculate the range of possible reflectance factors from the provided valid range of radiance, and use it to normalize before the gamma correction
                rf_min, rf_max = (
                    self.rad_range * np.pi * np.square(self.meta.instrument_meta.esd)
                ) / self.meta.instrument_meta.esun
                self._bv = abi.rf2bv(self.cmi, min=rf_min, max=rf_max, gamma=self.gamma)

            elif 7 <= self.meta.instrument_meta.band_id <= 16:
                self._bv = abi.bt2bv(self.cmi)

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
        r_coeff=0.45,
        g_coeff=0.1,
        b_coeff=0.45,
        upscale=False,
        gamma=1.0,
        mask_fill=False,
    ):
        """
        Creates the "natural" color RGB for ABI following https://doi.org/10.1029/2018EA000379 in BGR order

        Arguments:
            - `red_nc`, `green_nc`, `blue_nc`: Strings or Path objects pointing to GOES-R ABI L1b Radiance netCDF files for the red (0.64 μm), green (0.86 μm), and blue (0.47 μm) components
            - `r_coeff`, `g_coeff`, `b_coeff`: Coefficients for the fractional combination "green" band method described in Bah et. al (2018)
            - `upscale`: Whether to scale up green and blue images (1 km) to match the red image (500 m) (`True`) or vice versa (`False`, Default)
            - `gamma`: Optional gamma correction for reflective ABI brightness value. Defaults to no correction
            - `mask_fill` Optionally fills masked radiance with np.nan and masked DQF with 0. Default `False`
        """

        super(ABINaturalRGB, self).__init__()

        red_image = ABIImage(red_nc, gamma=gamma, mask_fill=mask_fill)
        green_image = ABIImage(green_nc, gamma=gamma, mask_fill=mask_fill)
        blue_image = ABIImage(blue_nc, gamma=gamma, mask_fill=mask_fill)

        if upscale:
            # upscale green and blue to the size of red
            green_image.bv = cv2.resize(
                green_image.bv,
                (red_image.bv.shape[1], red_image.bv.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            green_image.dqf = cv2.resize(
                green_image.bv,
                (red_image.dqf.shape[1], red_image.dqf.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            blue_image.bv = cv2.resize(
                blue_image.bv,
                (red_image.bv.shape[1], red_image.bv.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            blue_image.dqf = cv2.resize(
                blue_image.bv,
                (red_image.dqf.shape[1], red_image.dqf.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        else:
            # downscale red to the size of green and blue
            red_image.bv = cv2.resize(
                red_image.bv,
                (green_image.bv.shape[1], green_image.bv.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            red_image.dqf = cv2.resize(
                red_image.bv,
                (green_image.dqf.shape[1], green_image.dqf.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        green_image.bv = (
            (red_image.bv * r_coeff)
            + (green_image.bv * g_coeff)
            + (blue_image.bv * b_coeff)
        )

        self.bv = util.make_8bit(
            np.stack([blue_image.bv, green_image.bv, red_image.bv], axis=2)
        )
        self.quality = (
            sum([red_image.quality, green_image.quality, blue_image.quality]) / 3
        )
        self.dqf = np.stack([blue_image.dqf, green_image.dqf, red_image.dqf], axis=2)

        if upscale:
            self.meta = red_image.meta

        else:
            self.meta = green_image.meta

        self.meta.instrument_meta.band_id = (
            self.meta.instrument_meta.band_id_safe
        ) = "Color"
        self.meta.dataset_name = "RGB from " + ", ".join(
            (str(red_nc), str(green_nc), str(blue_nc))
        )
        self.meta.instrument_meta.rgb = [str(red_nc), str(green_nc), str(blue_nc)]


class SUVIImage(Image):
    def __init__(
        self,
        suvi_nc,
        shift=True,
        flip=True,
        dqf_correction=True,
        mask_fill=False,
    ):
        """
        Creates a 1-second 8-bit SUVI image made to look similar to what is shown on the SWPC website: https://www.swpc.noaa.gov/products/goes-solar-ultraviolet-imager-suvi

        Arguments:
            - `suvi_nc`: String or Path object pointing to a 1-second exposure GOES-R SUVI L1b Solar Imagery netCDF file
            - `shift`: Whether to try moving the center of the Sun to the center of the image. Default `True`
            - `flip`: Whether to flip the SUVI image from S-N to N-S to match SWPC. Default `True`
            - `dqf_correction`: Whether to interpolate over bad pixels marked by DQF. Default `True`
            - `mask_fill` Optionally fills masked radiance with np.nan and masked DQF with 0. Default `False`
        """

        super(SUVIImage, self).__init__()

        with netCDF4.Dataset(suvi_nc, "r") as loaded_nc:
            self.rad = loaded_nc["RAD"][:]
            # count of non-masked elements over total number of elements
            self.quality = self.rad.count() / self.rad.size
            self.dqf = loaded_nc["DQF"][:]
            x_offset = 640 - loaded_nc["CRPIX1"][:]
            y_offset = 640 - loaded_nc["CRPIX2"][:]

        if mask_fill:
            self.rad[self.rad.mask] = np.nan
            self.dqf[self.dqf.mask] = 0

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
            # move the sun to the center of the image using default ndimage spline interpolation
            # if the sun is more than 100 pixels away from center in both axes, do nothing
            if abs(x_offset) < 100 and abs(y_offset) < 100:
                self.rad = ndimage.shift(self.rad, (y_offset, x_offset), mode="wrap")

        if flip:
            # SUVI arrays are S-N, make them N-S to match SWPC
            self.rad = np.flipud(self.rad)

        self.meta = meta.NCMeta(suvi_nc)

    @property
    def bv(self):
        if self._bv is None:
            self._bv = suvi.rad_bv(
                self.rad,
                *self.meta.instrument_meta.coefficients.input_range,
                self.meta.instrument_meta.coefficients.asinh_a,
                *self.meta.instrument_meta.coefficients.output_range,
            )

        return self._bv
