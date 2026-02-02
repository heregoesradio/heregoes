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

import datetime

import numpy as np

from heregoes.core import heregoes_njit, heregoes_njit_noparallel


@heregoes_njit
def navigate(y_rad, x_rad, lon_origin, r_eq, r_pol, sat_height):  # pragma: no cover
    """
    Given instrument scanning angles `y_rad`, `x_rad`,\n
    longitude of projection origin `lon_origin`,\n
    semi-major axis `r_eq`, semi-minor axis `r_pol`,\n
    and the perspective point height `sat_height`,\n
    return ellipsoidal latitude and longitude in degrees\n
    following 7.1.2.8.1 in the PUG Volume 4:
    https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf
    """
    H = sat_height + r_eq

    lambda_0 = np.deg2rad(lon_origin)

    a = np.square(np.sin(x_rad)) + (
        np.square(np.cos(x_rad))
        * (
            np.square(np.cos(y_rad))
            + (((np.square(r_eq)) / (np.square(r_pol))) * np.square(np.sin(y_rad)))
        )
    )
    b = -2.0 * H * np.cos(x_rad) * np.cos(y_rad)
    c = np.square(H) - np.square(r_eq)

    r_s = (-b - np.sqrt((np.square(b)) - (4.0 * a * c))) / (2.0 * a)

    s_x = r_s * np.cos(x_rad) * np.cos(y_rad)
    s_y = -r_s * np.sin(x_rad)
    s_z = r_s * np.cos(x_rad) * np.sin(y_rad)

    lat_deg = np.rad2deg(
        np.arctan(
            ((np.square(r_eq)) / (np.square(r_pol)))
            * ((s_z / np.sqrt((np.square((H - s_x))) + (np.square(s_y)))))
        )
    )
    lon_deg = np.rad2deg(lambda_0 - np.arctan(s_y / (H - s_x)))

    return (
        np.atleast_1d(lat_deg).astype(np.float32),
        np.atleast_1d(lon_deg).astype(np.float32),
    )


@heregoes_njit
def inverse_navigate(
    lat_deg, lon_deg, lon_origin, r_eq, r_pol, sat_height, feature_height
):  # pragma: no cover
    """
    Given ellipsoidal latitude `lat_deg` and longitude `lon_deg`,\n
    longitude of projection origin `lon_origin`,\n
    semi-major axis `r_eq`, semi-minor axis `r_pol`,\n
    the perspective point height `sat_height`,\n
    and the ellipsoidal height of the feature (typically cloud or terrain) in meters `feature_height`,\n
    return instrument scanning angles `y_rad`, `x_rad`\n
    following 7.1.2.8.2 in the PUG Volume 4:
    https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf

    """
    phi = np.deg2rad(lat_deg)
    lambda_ = np.deg2rad(lon_deg)

    e = 0.0818191910435

    H = sat_height + r_eq

    lambda_0 = np.deg2rad(lon_origin)

    phi_c = np.arctan((np.square(r_pol) / np.square(r_eq)) * np.tan(phi))

    r_c = r_pol / np.sqrt(1 - np.square(e) * np.square(np.cos(phi_c)))
    r_c += feature_height  # change the height above the ellipsoid

    s_x = H - r_c * np.cos(phi_c) * np.cos(lambda_ - lambda_0)
    s_y = -r_c * np.cos(phi_c) * np.sin(lambda_ - lambda_0)
    s_z = r_c * np.sin(phi_c)

    y_rad = np.arctan(s_z / s_x)
    x_rad = np.arcsin(-s_y / np.sqrt(np.square(s_x) + np.square(s_y) + np.square(s_z)))

    return (
        np.atleast_1d(y_rad).astype(np.float32),
        np.atleast_1d(x_rad).astype(np.float32),
    )


@heregoes_njit
def pixel_height(y_rad, r_eq, sat_height, ifov):  # pragma: no cover
    # https://doi.org/10.1017/CBO9781139029346.005 eqs. 3.10 a-b
    # cross-track (N-S distance for ABI)

    r = r_eq
    sh = sat_height
    beta = ifov

    alpha_c = y_rad
    delta = np.arcsin(((sh + r) / r) * np.sin(alpha_c)) - alpha_c
    Lc = 2 * (((r * np.sin(delta)) / np.sin(alpha_c)) * np.tan(beta / 2.0))

    return np.atleast_1d(Lc).astype(np.float32)


@heregoes_njit
def pixel_width(x_rad, r_eq, sat_height, ifov):  # pragma: no cover
    # https://doi.org/10.1017/CBO9781139029346.005 eqs. 3.7 - 3.9
    # along-track (W-E distance for ABI)

    r = r_eq
    sh = sat_height
    beta = ifov

    alpha_a = x_rad
    alpha_a1 = alpha_a - beta / 2.0
    alpha_a2 = alpha_a + beta / 2.0
    L1 = r * (np.arcsin(((sh + r) / r) * np.sin(alpha_a1)) - alpha_a1)
    L2 = r * (np.arcsin(((sh + r) / r) * np.sin(alpha_a2)) - alpha_a2)
    La = L2 - L1

    return np.atleast_1d(La).astype(np.float32)


@heregoes_njit_noparallel
def pixel_area(cross_track, along_track):  # pragma: no cover
    return np.atleast_1d(
        cross_track[:, np.newaxis] * along_track[np.newaxis, :]
    ).astype(np.float32)


def fractional_jd(utc_time: datetime.datetime) -> float:
    # returns fractional days since J2000 epoch
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
    return (utc_time - j2000).total_seconds() / 86400


@heregoes_njit
def norm_az(az_rad, degrees=False):  # pragma: no cover
    """
    Given azimuth in radians, normalize to between 0 and 2pi (in degrees if `degrees` is True)
    """
    _norm_az = np.atleast_1d((az_rad + (2.0 * np.pi)) % (2.0 * np.pi)).astype(
        np.float32
    )
    if degrees:
        _norm_az = np.rad2deg(_norm_az)
    return _norm_az


@heregoes_njit
def el2za(el_rad, degrees=False):  # pragma: no cover
    """
    Given elevation angle above horizon in radians, return zenith angle (in degrees if `degrees` is True)
    """
    za_rad = np.atleast_1d(np.pi / 2.0 - el_rad).astype(np.float32)
    if degrees:
        za_rad = np.rad2deg(za_rad)
    return za_rad


@heregoes_njit
def za2el(za_rad, degrees=False):  # pragma: no cover
    """
    Given zenith angle in radians, return elevation angle above horizon (in degrees if `degrees` is True)
    """
    el_rad = np.atleast_1d(np.pi / 2.0 - za_rad).astype(np.float32)
    if degrees:
        el_rad = np.rad2deg(el_rad)
    return el_rad
