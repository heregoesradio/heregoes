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

"""Storage for instrument-specific coefficients not delivered within their netCDF products"""

import re


class ABICoeff:
    def __init__(self, platform_id, band_id):
        """
        Stores instrument-specific band equivalent widths as (wavenumber, wavelength) (cm^-1, μm) for ABI
        """

        # https://cimss.ssec.wisc.edu/goes/calibration/eqw/GOES16_ABI_ALLBANDS_MAR2016.eqw
        abi_g16_eqw = {
            1: (1695.3619, 0.0376),
            2: (2028.3127, 0.0826),
            3: (464.8830, 0.0347),
            4: (72.5596, 0.0137),
            5: (174.3903, 0.0452),
            6: (91.7739, 0.0462),
            7: (116.4128, 0.1763),
            8: (202.0661, 0.7728),
            9: (86.1816, 0.4140),
            10: (34.7444, 0.1870),
            11: (57.9918, 0.4139),
            12: (40.8063, 0.3768),
            13: (29.7437, 0.3176),
            14: (61.5342, 0.7711),
            15: (59.6483, 0.8989),
            16: (30.8270, 0.5428),
        }

        # https://cimss.ssec.wisc.edu/goes/calibration/eqw/GOES17_ABI_ALLBANDS_MAR2016.eqw
        abi_g17_eqw = {
            1: (1682.6986, 0.0373),
            2: (2078.7224, 0.0844),
            3: (462.0279, 0.0345),
            4: (72.5726, 0.0137),
            5: (174.2840, 0.0451),
            6: (91.5246, 0.0460),
            7: (125.9091, 0.1902),
            8: (201.5474, 0.7708),
            9: (87.6595, 0.4217),
            10: (34.2198, 0.1842),
            11: (57.1147, 0.4085),
            12: (41.2183, 0.3809),
            13: (30.5115, 0.3251),
            14: (60.0778, 0.7537),
            15: (59.4295, 0.8970),
            16: (30.7490, 0.5441),
        }

        # https://cimss.ssec.wisc.edu/goes/calibration/eqw/GOES-T_ABI_ALLBANDS_MAR2016.eqw
        abi_g18_eqw = {
            1: (1674.7174, 0.0371),
            2: (2049.2712, 0.0833),
            3: (463.8655, 0.0346),
            4: (72.5934, 0.0137),
            5: (174.3004, 0.0451),
            6: (91.5551, 0.0460),
            7: (126.9894, 0.1935),
            8: (212.4962, 0.8167),
            9: (80.7394, 0.3890),
            10: (33.1602, 0.1788),
            11: (58.9062, 0.4230),
            12: (40.1364, 0.3693),
            13: (29.9888, 0.3199),
            14: (61.0177, 0.7692),
            15: (60.4056, 0.9080),
            16: (31.1519, 0.5486),
        }

        # https://cimss.ssec.wisc.edu/goes/calibration/eqw/GOES-U_ABI_ALLBANDS_MAR2016.eqw
        abi_g19_eqw = {
            1: (1696.9133, 0.0376),
            2: (2031.9250, 0.0827),
            3: (466.4469, 0.0348),
            4: (84.0518, 0.0160),
            5: (173.2802, 0.0448),
            6: (91.8146, 0.0462),
            7: (132.9054, 0.2027),
            8: (213.0019, 0.8176),
            9: (80.7291, 0.3893),
            10: (33.7350, 0.1814),
            11: (48.6344, 0.3510),
            12: (41.3170, 0.3799),
            13: (28.9552, 0.3081),
            14: (58.9590, 0.7393),
            15: (60.8712, 0.9141),
            16: (30.2630, 0.5341),
        }

        if bool(re.search("g.*16", str(platform_id).lower())):
            self.eqw = abi_g16_eqw[band_id]

        elif bool(re.search("g.*17", str(platform_id).lower())):
            self.eqw = abi_g17_eqw[band_id]

        elif bool(re.search("g.*18", str(platform_id).lower())):
            self.eqw = abi_g18_eqw[band_id]

        elif bool(re.search("g.*19", str(platform_id).lower())):
            self.eqw = abi_g19_eqw[band_id]

        else:
            raise NotImplementedError(
                f"No ABI coefficients for platform {platform_id}."
            )


class SUVICoeff:
    def __init__(self, wavelength):
        """
        Stores scaling coefficients used for SUVI images at (Here GOES Radiotelescope)[heregoesradio.com].
        The coefficients were chosen in 2020-2021 to approximate the appearance of SWPC G16 SUVI imagery from 1-second exposures.
        See the SUVIImage class for how they are implemented.
        """

        input_range = {
            94: (0.0, 40.0),
            131: (0.0, 40.0),
            171: (0.0, 20.0),
            195: (0.0, 50.0),
            284: (0.0, 60.0),
            304: (0.0, 300.0),
        }

        asinh_a = {
            94: 0.001,
            131: 0.00015,
            171: 0.0015,
            195: 0.0015,
            284: 0.00025,
            304: 0.00015,
        }

        output_range = {
            94: (0.0, 0.8),
            131: (0.117, 0.8),
            171: (0.0, 1.0),
            195: (0.117, 1.0),
            284: (0.2, 0.8),
            304: (0.117, 0.8),
        }

        self.input_range = input_range[wavelength]
        self.asinh_a = asinh_a[wavelength]
        self.output_range = output_range[wavelength]
