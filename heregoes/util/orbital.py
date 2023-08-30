# Copyright (c) 2011, 2012, 2013, 2014, 2015.

# Author(s):

#   Esben S. Nielsen <esn@dmi.dk>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""
Contains orbital and astronomy routines copied from Pyorbital v1.7.1 (GPLv3) (https://github.com/pytroll/pyorbital)
and modified to use the Numba @njit compiler
"""

import datetime

import numpy as np

from heregoes.util import njit

F = 1 / 298.257223563  # Earth flattening WGS-84
A = 6378.137  # WGS84 Equatorial radius
MFACTOR = 7.292115e-5


###########################################################################
#############################Here GOES helpers#############################


def jdays2000(utc_time):
    # returns fractional days since J2000 epoch
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
    return (utc_time - j2000).total_seconds() / 86400


@njit.heregoes_njit_noparallel
def norm_az(az_rad):
    return np.atleast_1d((az_rad + (2.0 * np.pi)) % (2.0 * np.pi)).astype(np.float32)


@njit.heregoes_njit_noparallel
def el2za(el_rad):
    return np.atleast_1d(np.pi / 2.0 - el_rad).astype(np.float32)


@njit.heregoes_njit_noparallel
def za2el(za_rad):
    return np.atleast_1d(np.pi / 2.0 - za_rad).astype(np.float32)


###########################################################################
##########################Modified Pyorbital code##########################


@njit.heregoes_njit_noparallel
# https://github.com/pytroll/pyorbital/blob/v1.7.1/pyorbital/astronomy.py#L110
def _local_hour_angle(jdays2000, longitude, right_ascension):
    """Hour angle at *utc_time* for the given *longitude* and
    *right_ascension*
    longitude in radians
    """
    return _lmst(jdays2000, longitude) - right_ascension


@njit.heregoes_njit_noparallel
# https://github.com/pytroll/pyorbital/blob/v1.7.1/pyorbital/astronomy.py#L66
def _lmst(jdays2000, longitude):
    """Local mean sidereal time, computed from *utc_time* and *longitude*.
    In radians.
    """
    return gmst(jdays2000) + longitude


@njit.heregoes_njit_noparallel
# https://github.com/pytroll/pyorbital/blob/v1.7.1/pyorbital/astronomy.py#L54
def gmst(jdays2000):
    """Greenwich mean sidereal utc_time, in radians.
    As defined in the AIAA 2006 implementation:
    http://www.celestrak.com/publications/AIAA/2006-6753/
    """
    ut1 = jdays2000 / 36525.0
    theta = 67310.54841 + ut1 * (
        876600 * 3600.0 + 8640184.812866 + ut1 * (0.093104 - ut1 * 6.2 * 10.0e-6)
    )
    return np.deg2rad(theta / 240.0) % (2.0 * np.pi)


@njit.heregoes_njit_noparallel
# https://github.com/pytroll/pyorbital/blob/v1.7.1/pyorbital/astronomy.py#L91
def sun_ra_dec(jdays2000):
    """Right ascension and declination of the sun at *utc_time*."""
    jdate = jdays2000 / 36525.0
    eps = np.deg2rad(
        23.0
        + 26.0 / 60.0
        + 21.448 / 3600.0
        - (46.8150 * jdate + 0.00059 * jdate * jdate - 0.001813 * jdate * jdate * jdate)
        / 3600.0
    )
    eclon = sun_ecliptic_longitude(jdays2000)
    x__ = np.cos(eclon)
    y__ = np.cos(eps) * np.sin(eclon)
    z__ = np.sin(eps) * np.sin(eclon)
    r__ = np.sqrt(1.0 - z__ * z__)
    # sun declination
    declination = np.arctan2(z__, r__)
    # right ascension
    right_ascension = 2.0 * np.arctan2(y__, (x__ + r__))
    return right_ascension, declination


@njit.heregoes_njit_noparallel
# https://github.com/pytroll/pyorbital/blob/v1.7.1/pyorbital/astronomy.py#L73
def sun_ecliptic_longitude(jdays2000):
    """Ecliptic longitude of the sun at *utc_time*."""
    jdate = jdays2000 / 36525.0
    # mean anomaly, rad
    m_a = np.deg2rad(
        357.52910
        + 35999.05030 * jdate
        - 0.0001559 * jdate * jdate
        - 0.00000048 * jdate * jdate * jdate
    )
    # mean longitude, deg
    l_0 = 280.46645 + 36000.76983 * jdate + 0.0003032 * jdate * jdate
    d_l = (
        (1.914600 - 0.004817 * jdate - 0.000014 * jdate * jdate) * np.sin(m_a)
        + (0.019993 - 0.000101 * jdate) * np.sin(2.0 * m_a)
        + 0.000290 * np.sin(3.0 * m_a)
    )
    # true longitude, deg
    l__ = l_0 + d_l
    return np.deg2rad(l__)


@njit.heregoes_njit_noparallel
# https://github.com/pytroll/pyorbital/blob/v1.7.1/pyorbital/astronomy.py#L174
def observer_position(jdays2000, lon, lat, alt):
    """Calculate observer ECI position.
    http://celestrak.com/columns/v02n03/
    """

    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    theta = (gmst(jdays2000) + lon) % (2.0 * np.pi)
    c = 1.0 / np.sqrt(1.0 + F * (F - 2.0) * np.sin(lat) ** 2.0)
    sq = c * (1.0 - F) ** 2.0

    achcp = (A * c + alt) * np.cos(lat)
    x = achcp * np.cos(theta)  # kilometers
    y = achcp * np.sin(theta)
    z = (A * sq + alt) * np.sin(lat)

    vx = -MFACTOR * y  # kilometers/second
    vy = MFACTOR * x
    vz = 0.0

    return (x, y, z), (vx, vy, vz)


@njit.heregoes_njit_noparallel
# https://github.com/pytroll/pyorbital/blob/v1.7.1/pyorbital/astronomy.py#L118
def get_alt_az(jdays2000, lon, lat):
    """Return sun altitude and azimuth from *utc_time*, *lon*, and *lat*.
    lon,lat in degrees
    The returned angles are given in radians.
    """
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    ra_, dec = sun_ra_dec(jdays2000)
    h__ = _local_hour_angle(jdays2000, lon, ra_)

    alt = np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(h__))

    az = np.arctan2(
        -np.sin(h__), (np.cos(lat) * np.tan(dec) - np.sin(lat) * np.cos(h__))
    )

    return alt, az


@njit.heregoes_njit_noparallel
# https://github.com/pytroll/pyorbital/blob/v1.7.1/pyorbital/orbital.py#L244
def get_observer_look(sat_lon, sat_lat, sat_alt, jdays2000, lon, lat, alt):
    """Calculate observers look angle to a satellite.
    http://celestrak.com/columns/v02n02/
    :utc_time: Observation time (datetime object)
    :lon: Longitude of observer position on ground in degrees east
    :lat: Latitude of observer position on ground in degrees north
    :alt: Altitude above sea-level (geoid) of observer position on ground in km
    :return: (Azimuth, Elevation)
    """
    (pos_x, pos_y, pos_z), (vel_x, vel_y, vel_z) = observer_position(
        jdays2000, sat_lon, sat_lat, sat_alt
    )

    (opos_x, opos_y, opos_z), (ovel_x, ovel_y, ovel_z) = observer_position(
        jdays2000, lon, lat, alt
    )

    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    theta = (gmst(jdays2000) + lon) % (2.0 * np.pi)

    rx = pos_x - opos_x
    ry = pos_y - opos_y
    rz = pos_z - opos_z

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    top_s = sin_lat * cos_theta * rx + sin_lat * sin_theta * ry - cos_lat * rz
    top_e = -sin_theta * rx + cos_theta * ry
    top_z = cos_lat * cos_theta * rx + cos_lat * sin_theta * ry + sin_lat * rz

    # Azimuth is undefined when elevation is 90 degrees, 180 (pi) will be returned.
    az_ = np.arctan2(-top_e, top_s) + np.pi
    az_ = np.mod(az_, 2.0 * np.pi)  # Needed on some platforms

    rg_ = np.sqrt(rx * rx + ry * ry + rz * rz)

    top_z_divided_by_rg_ = top_z / rg_

    # Due to rounding top_z can be larger than rg_ (when el_ ~ 90).
    top_z_divided_by_rg_ = np.clip(
        top_z_divided_by_rg_, np.nanmin(top_z_divided_by_rg_), 1.0
    )
    el_ = np.arcsin(top_z_divided_by_rg_)

    return az_, el_
