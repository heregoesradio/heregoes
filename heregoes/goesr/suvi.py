# Copyright (c) 2020-2023.

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

"""Functions for working with SUVI L1b Solar Imagery"""

import numpy as np

from heregoes.core import heregoes_njit
from heregoes.util import linear_norm, make_8bit


@heregoes_njit
def rad2bv(
    radiance, input_min, input_max, asinh_a, output_min, output_max
):  # pragma: no cover
    # rescale radiance between 0 and 1 based on the estimated minimum and maximum possible values of radiance for a given channel
    rad = linear_norm(
        radiance, old_min=input_min, old_max=input_max, new_min=0.0, new_max=1.0
    )
    rad = np.clip(rad, 0.0, 1.0)

    # perform a non-linear transform between linear stretches
    rad = np.arcsinh(rad / asinh_a) / np.arcsinh(1 / asinh_a)

    # rescale transformed radiance for better contrast
    rad = linear_norm(
        rad, old_min=output_min, old_max=output_max, new_min=0.0, new_max=1.0
    )
    rad = np.clip(rad, 0.0, 1.0)

    return make_8bit(rad * 255)
