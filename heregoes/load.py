"""
Loads a supported GOES-R ABI or SUVI netCDF file if not already loaded
"""

from heregoes.core.types import GOESRInputType
from heregoes.goesr import ABIL1bData, ABIL2Data, GOESRData, SUVIL1bData


def load(nc_data: GOESRInputType) -> GOESRData:
    """
    Returns a walkable interface to netCDF attributes, dimensions, and variables

    #### Parameters:
    - `nc_data`: a `str` or `PathLike` object pointing to a GOES-R netCDF file
    """
    if isinstance(nc_data, GOESRData):
        return nc_data

    match str(nc_data).lower():
        case s if "abi-l1b" in s:
            return ABIL1bData(nc_data)

        case s if "abi-l2" in s:
            return ABIL2Data(nc_data)

        case s if "suvi-l1b" in s:
            return SUVIL1bData(nc_data)

        case _:
            raise ValueError(f"Product type is not supported: {nc_data}.")
