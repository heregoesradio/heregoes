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

import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy import ndimage

from heregoes import load
from heregoes.core.types import SUVIInputType
from heregoes.goesr import suvi
from heregoes.image._image import _Image
from heregoes.util import make_8bit, make_rgb, max_time_delta

logger = logging.getLogger("heregoes-logger")
safe_time_format = "%Y-%m-%dT%H%M%SZ"


class SUVIImage(_Image):
    """
    ### SUVI L1b radiance imagery with DQF correction

    Creates a 1-second exposure 8-bit SUVI image made to look similar to what is shown on the SWPC website:
    https://www.swpc.noaa.gov/products/goes-solar-ultraviolet-imager-suvi

    We adopt the same inverse hyperbolic sine enhancement as in `astropy.visualization.stretch.AsinhStretch`:

    `asinh(rad / α) / asinh(1 / α)`

    where the alpha term is given by the argument `asinh_a`, and rad is SUVI radiance normalized to `input_range`.
    A second linear normalization occurs after the asinh stretch using `output_range`.

    If not provided as arguments, `input_range`, `asinh_a`, and `output_range` use the default per-channel coefficients defined in `heregoes.goesr.coefficients.SUVICoeff`.

    ### Parameters:
        - `suvi_data`:
            - Either a str or Path referencing a 1-second exposure SUVI L1b Solar Imagery netCDF file,
            - or the `SUVIL1bData` object formed by `heregoes.load()` on the path

        - `shift` (optional):
            - Whether to try moving the center of the Sun to the center of the image. Default `True`

        - `shift_limit` (optional):
            - Limits the shift operation that moves the Sun to the center of the image to a maximum of `shift_limit` pixels. Default `100`

        - `flip` (optional):
            - Whether to flip the SUVI image from S-N to N-S to match SWPC. Default `True`

        - `dqf_correction` (optional):
            - Whether to interpolate over bad pixels marked by DQF. Default `True`

        - `input_range` (optional):
            - A tuple of the (min, max) floating point radiance used to normalize the input image

        - `asinh_a` (optional):
            - The alpha term of the hyperbolic arcsine stretch

        - `output_range` (optional):
            - A tuple of the (min, max) normalized brightness value (0.0...1.0) at which to clip the output image

    ### Attributes:
        - `rad`:
            - Radiance in units `W m-2 sr-1`

        - `dqf`:
            - Data Quality Flag array delivered with the product

        - `quality`:
            - The ratio of masked to total image array elements

        - `bv` (Brightness Value):
            - Pixel values 0-255
    """

    def __init__(
        self,
        suvi_data: SUVIInputType,
        shift: bool = True,
        shift_limit: int = 100,
        flip: bool = True,
        dqf_correction: bool = True,
        input_range: Optional[tuple[float, float]] = None,
        asinh_a: Optional[float] = None,
        output_range: Optional[tuple[float, float]] = None,
    ):
        self._bv = None

        self.suvi_data = load(suvi_data)

        self.time = np.datetime64(
            self.suvi_data.time_coverage_start.replace(tzinfo=None),
        )

        if self.suvi_data["CMD_EXP"][...] != 1.0:
            logger.warning(
                f"Short SUVI exposure detected in {self.suvi_data.filepath}:\nSUVI exposures shorter than 1 second are not officially supported.",
            )

        self.input_range = (
            input_range
            if input_range
            else self.suvi_data.instrument_coefficients.input_range
        )
        self.asinh_a = (
            asinh_a if asinh_a else self.suvi_data.instrument_coefficients.asinh_a
        )
        self.output_range = (
            output_range
            if output_range
            else self.suvi_data.instrument_coefficients.output_range
        )

        self.rad = self.suvi_data["RAD"][...]
        self.dqf = self.suvi_data["DQF"][...]
        self.quality = self.suvi_data["RAD"].pct_unmasked
        x_offset = 640 - self.suvi_data["CRPIX1"][...]
        y_offset = 640 - self.suvi_data["CRPIX2"][...]

        self.default_filename = "_".join(
            (
                self.suvi_data.platform_ID.lower(),
                self.suvi_data._instrument_type_str.lower(),
                self.suvi_data._wavelength_str.lower(),
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
            # move the sun to the center of the image to a maximum of `shift_limit` pixels
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
                *self.input_range,
                self.asinh_a,
                *self.output_range,
            )

        return self._bv

    @bv.setter
    def bv(self, value):
        self._bv = value


class SUVIRGB(_Image):
    """
    ### Creates a custom SUVI RGB from three SUVIImage objects

    ### Parameters:
        - `red_image`, `green_image`, `blue_image`:
            - SUVIImage objects to use for the red, green, and blue channels

    ### Attributes:
        - `quality`:
            - The ratio of masked to total image pixels across red, green, and blue components

        - `bv` (Brightness Value):
            - Color pixel values 0-255 with dimensions in BGR order
    """

    def __init__(
        self, red_image: SUVIImage, green_image: SUVIImage, blue_image: SUVIImage
    ):
        self.suvi_data = red_image.suvi_data

        self.time = np.array(
            [
                red_image.time,
                green_image.time,
                blue_image.time,
            ],
            dtype="datetime64",
        )

        max_delta_m = max_time_delta(self.time) / 60
        if max_delta_m > 20:
            logger.warning(
                f"SUVI images taken outside of a 20-minute window will blur the RGB."
            )

        self.bv = make_8bit(
            make_rgb(
                red_image.bv,
                green_image.bv,
                blue_image.bv,
            )
        )

        self.quality = (
            sum([red_image.quality, green_image.quality, blue_image.quality]) / 3
        )

        # update some meta to reflect that this is an RGB
        self.suvi_data["WAVELNTH"][...] = np.atleast_1d(0)
        self.suvi_data._wavelength_str = "Color"

        self.suvi_data.rgb = [
            str(red_image.suvi_data),
            str(green_image.suvi_data),
            str(blue_image.suvi_data),
        ]
        self.suvi_data.dataset_name = "RGB from " + ", ".join(self.suvi_data.rgb)

        self.default_filename = "_".join(
            (
                red_image.suvi_data.platform_ID.lower(),
                red_image.suvi_data._instrument_type_str.lower(),
                self.suvi_data._wavelength_str.lower(),
                red_image.suvi_data.time_coverage_start.strftime(safe_time_format),
            )
        )
