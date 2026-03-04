from os import PathLike
from pathlib import Path

import cv2
import numpy as np
from srtm import get_ellipsoidal_srtm
from figure import index_delta_figure, latlon_target

from heregoes.image import ABIImage
from heregoes.projection import ABIProjection
from heregoes.util import (
    make_8bit,
    minmax,
    x2,
)
from tests.resources_l1b import abi_cc02_nc

"""Orthorectification of the ABI Fixed Grid with heregoes (netCDF4, Numpy, GDAL, cv2)"""

# in a ~250x250 px region of the Cascade Range in the GOES-East CONUS,
lat_bounds = 47.48457, 44.93645
lon_bounds = -125.14742, -118.879265

# we will track displacement by parallax of these mountain summit coordinates taken from SRTM15
target_latlon = [
    (46.852085, -121.760414),  # Mt. Rainier  4317.157 m
    (46.189583, -122.18958),  # Mt. St. Helens  2396.208 m
    (46.202084, -121.49375),  # Mt. Adams  3650.2834 m
    (45.372917, -121.697914),  # Mt. Hood  3246.899 m
]

# Download GOES-{16...19} ABI C02 L1b netCDF from NOAA CLASS or AWS S3:
# https://noaa-goes16.s3.amazonaws.com/index.html
abi_nc_path = abi_cc02_nc
abi_gamma = 2 / 3

# Download SRTM15 netCDF from:
# https://topex.ucsd.edu/WWW_html/srtm15_plus.html
srtm_nc_path = "/home/wx-star/geo/SRTM/SRTM15_V2.7.nc"

image_path = Path("img")
image_path.mkdir(exist_ok=True)


def ortho_abi_from_srtm15(
    abi_nc_path: PathLike,
    srtm_nc_path: PathLike,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    target_latlon: list[tuple[float, float]],
    abi_gamma: float,
    image_path: PathLike,
    upscale_factor=2,
):
    """Visualize forward and backward orthorectification of ABI using SRTM15 netCDF"""

    # get 15-arcsecond SRTM height data for the ABI region
    srtm_lat, srtm_lon, srtm_height = get_ellipsoidal_srtm(
        srtm_nc_path=srtm_nc_path,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )
    srtm_lat_bounds = srtm_lat[0, 0], srtm_lat[-1, -1]
    srtm_lon_bounds = srtm_lon[0, 0], srtm_lon[-1, -1]

    # SRTM15 includes bathymetry, so only look down to 0 m
    srtm_height = np.where(srtm_height > 0, srtm_height, 0)

    # form ABIProjection object subsetted on the ABI bounds using SRTM heights to parallax-correct
    abi_projection = ABIProjection(
        abi_nc_path,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        height_m=[srtm_height[0, 0], srtm_height[-1, -1]],
    )

    # warp SRTM height data to the projection of the ABI scene
    warped_heights = abi_projection.resample2abi(
        srtm_height,
        lat_bounds=srtm_lat_bounds,
        lon_bounds=srtm_lon_bounds,
        resample_algo="cubic",
    )

    # make new images using ABI Fixed Grid lat/lon and the warped SRTM heights

    # if resample_nav=False (default), the ABI image is orthorectified to correct for terrain parallax
    img_resampled_to_nav = ABIImage(
        abi_nc_path,
        gamma=abi_gamma,
        lat_bounds=abi_projection.lat_deg,
        lon_bounds=abi_projection.lon_deg,
        height_m=warped_heights,
        resample_nav=False,
    )

    # if resample_nav=True, the underlying navigation is orthorectified and image pixels are untouched
    nav_resampled_to_img = ABIImage(
        abi_nc_path,
        gamma=abi_gamma,
        lat_bounds=abi_projection.lat_deg,
        lon_bounds=abi_projection.lon_deg,
        height_m=warped_heights,
        resample_nav=True,
    )

    # render the uncorrected ABI image for comparison
    original_abi_img = ABIImage(
        abi_nc_path,
        gamma=abi_gamma,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )

    # save images with our target coordinates indicated in green
    cv2.imwrite(
        image_path.joinpath("original.png"),
        latlon_target(
            img=original_abi_img.bv,
            search_lat=original_abi_img.lat_deg,
            search_lon=original_abi_img.lon_deg,
            target_latlon=target_latlon,
            upscale_factor=upscale_factor,
        ),
    )

    cv2.imwrite(
        image_path.joinpath("resampled-image.png"),
        latlon_target(
            img=img_resampled_to_nav.bv,
            search_lat=img_resampled_to_nav.lat_deg,
            search_lon=img_resampled_to_nav.lon_deg,
            target_latlon=target_latlon,
            upscale_factor=upscale_factor,
        ),
    )
    cv2.imwrite(
        image_path.joinpath("resampled-nav.png"),
        latlon_target(
            img=nav_resampled_to_img.bv,
            search_lat=nav_resampled_to_img.lat_deg,
            search_lon=nav_resampled_to_img.lon_deg,
            target_latlon=target_latlon,
            upscale_factor=upscale_factor,
        ),
    )

    index_delta_figure(
        nav_resampled_to_img.index,
        nav_resampled_to_img.nav_index,
        filepath=image_path.joinpath(f"index-delta.png"),
    )

    # zero the inverse-orthorectified nav index,
    y, x = nav_resampled_to_img.nav_index
    norm_y = y - y.min()
    norm_x = x - x.min()
    norm_inverse_ortho_idx = np.clip(norm_y, 0, norm_y.shape[0] - 1), np.clip(
        norm_x, 0, norm_x.shape[1] - 1
    )

    # and use it for visualization
    warped_heights_img = make_8bit(minmax(warped_heights) * 255)
    warped_orthorectified_heights = make_8bit(
        minmax(warped_heights[norm_inverse_ortho_idx]) * 255
    )
    cv2.imwrite(
        image_path.joinpath("warped-heights.png"),
        x2(warped_heights_img),
    )
    cv2.imwrite(
        image_path.joinpath("warped-inverse-orthorectified-heights.png"),
        x2(warped_orthorectified_heights),
    )


if __name__ == "__main__":
    ortho_abi_from_srtm15(
        abi_nc_path=abi_nc_path,
        srtm_nc_path=srtm_nc_path,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        target_latlon=target_latlon,
        abi_gamma=abi_gamma,
        image_path=image_path,
    )
