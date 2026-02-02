# Copyright (c) 2025.

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
from typing import Annotated, Literal

from numpy.typing import NDArray

from heregoes.goesr import ABIL1bData, ABIL2Data, GOESRData, SUVIL1bData

# GOES-R inputs
GOESRInputType = GOESRData | PathLike | str
ABIL1bInputType = ABIL1bData | PathLike | str
ABIL2InputType = ABIL2Data | PathLike | str
ABIInputType = ABIL1bInputType | ABIL2InputType
SUVIInputType = SUVIL1bData | PathLike | str

# ABI Fixed Grid takes a continuous 2D index
_FixedGridIndexMemberType = int | slice | NDArray
FixedGridIndexType = tuple[_FixedGridIndexMemberType, _FixedGridIndexMemberType]

# each element of data on the Fixed Grid is either an array or scalar float, or length-2 tuples or lists thereof
_FixedGridDataMemberType = NDArray | float
FixedGridDataType = (
    _FixedGridDataMemberType
    | tuple[_FixedGridDataMemberType, _FixedGridDataMemberType]
    | Annotated[list[_FixedGridDataMemberType], 2]
)

__all__ = [
    "GOESRInputType",
    "ABIInputType",
    "SUVIInputType",
    "FixedGridIndexType",
    "FixedGridDataType",
]
