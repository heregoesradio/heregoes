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

"""Functions for working with ABI L1b Radiance and ABI L2 CMI"""

import numpy as np

from heregoes import heregoes_njit, util


@heregoes_njit
def rad2rf(rad, esd, esun):
    # converts spectral radiance "rad" to Reflectance Factor (RF) following CMIP ATBD eq. 3-4
    return ((rad * np.pi * np.square(esd)) / esun).astype(np.float32)


@heregoes_njit
def rf2rad(rf, esd, esun):
    # converts Reflectance Factor (RF) to spectral radiance "rad" following CMIP ATBD eq. 3-3
    return ((rf * esun) / (np.pi * np.square(esd))).astype(np.float32)


@heregoes_njit
def rf2bv(rf, min, max, gamma=1.0):
    # converts Reflectance Factor (RF) to an 8-bit representation (Brightness Value (BV)) following CMIP ATBD eq. 3-22
    return util.make_8bit(
        np.power(
            util.linear_norm(rf, old_min=min, old_max=max, new_min=0.0, new_max=1.0),
            gamma,
        )
        * 255
    )


@heregoes_njit
def rad2bt(rad, planck_fk1, planck_fk2, planck_bc1, planck_bc2):
    # converts spectral radiance "rad" to Brightness Temperature (BT) following CMIP ATBD eq. 3-5
    return (
        (planck_fk2 / (np.log((planck_fk1 / rad) + 1.0)) - planck_bc1) / planck_bc2
    ).astype(np.float32)


@heregoes_njit
def bt2rad(bt, planck_fk1, planck_fk2, planck_bc1, planck_bc2):
    # converts Brightness Temperature (BT) to spectral radiance "rad" following CMIP ATBD eq. 3-6
    return (
        planck_fk1 / (np.exp(planck_fk2 / (planck_bc1 + (planck_bc2 * bt))) - 1.0)
    ).astype(np.float32)


@heregoes_njit
def bt2bv(bt):
    # converts Brightness Temperature (BT) to an 8-bit representation (Brightness Value (BV)) following CMIP ATBD eq. 3-19 and 3-20
    return util.make_8bit(np.where(bt >= 242.0, (660.0 - (2.0 * bt)), (418.0 - bt)))


@heregoes_njit
def rad_wvn2wvl(rad, eqw_wvn, eqw_wvl):
    # converts spectral radiance "rad" in mW/m^2/sr/cm^-1 to W/m^2/sr/μm
    # https://cimss.ssec.wisc.edu/goes/calibration/Converting_AHI_RadianceUnits_24Feb2015.pdf
    return (rad / 1000.0 * eqw_wvn / eqw_wvl).astype(np.float32)


@heregoes_njit
def rad_wvl2wvn(rad, eqw_wvn, eqw_wvl):
    # converts spectral radiance "rad" in W/m^2/sr/μm to mW/m^2/sr/cm^-1
    # https://cimss.ssec.wisc.edu/goes/calibration/Converting_AHI_RadianceUnits_24Feb2015.pdf
    return (rad * 1000.0 / eqw_wvn * eqw_wvl).astype(np.float32)
