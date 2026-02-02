from pathlib import Path

import cv2
import numpy as np
from srtm import get_ellipsoidal_srtm

from heregoes.image import ABIImage
from heregoes.util import nearest_2d_search, nearest_scale, scale_idx
from tests.resources_l1b import abi_cc02_nc

"""Orthorectification of the ABI Fixed Grid with heregoes (netCDF4, Numpy, GDAL, cv2)"""

# in a ~500x500 px region of the Cascade Range in the GOES-East CONUS,
ul_lat, lr_lat = 47.48457, 44.93645
ul_lon, lr_lon = -125.14742, -118.879265

# we will track displacement by parallax of these mountain top coordinates
target_latlon = [
    (
        46.853322,
        -121.75991,
    ),  # Mt. Rainier http://www.ngs.noaa.gov/cgi-bin/ds_mark.prl?PidBox=SB1151
    (
        46.187639,
        -122.176594,
    ),  # Mt. St. Helens https://www.ngs.noaa.gov/cgi-bin/ds_mark.prl?PidBox=DH4126
    (
        46.202412,
        -121.490895,
    ),  # Mt. Adams https://www.ngs.noaa.gov/cgi-bin/ds_mark.prl?PidBox=SB1004
    (
        45.373514,
        -121.695919,
    ),  # Mt. Hood https://www.ngs.noaa.gov/cgi-bin/ds_mark.prl?PidBox=RC2244
]

# Download GOES-{16...19} ABI C02 L1b netCDF from NOAA CLASS or AWS S3:
# https://noaa-goes16.s3.amazonaws.com/index.html
abi_nc_path = abi_cc02_nc
abi_gamma = 2 / 3

# Download SRTM15 netCDF from:
# https://topex.ucsd.edu/WWW_html/srtm15_plus.html
srtm_nc_path = "/home/wx-star/geo/SRTM/SRTM15_V2.7.nc"


def demo():
    # render the uncorrected ABI image at our lat/lon bounding box,
    original_abi_img = ABIImage(
        abi_nc_path,
        gamma=abi_gamma,
        lat_bounds=[ul_lat, lr_lat],
        lon_bounds=[ul_lon, lr_lon],
    )

    # and record the upper left and lower right coordinates that were calculated on the ABI Fixed Grid
    abi_lat_bounds = (original_abi_img.lat_deg[0, 0], original_abi_img.lat_deg[-1, -1])
    abi_lon_bounds = (original_abi_img.lon_deg[0, 0], original_abi_img.lon_deg[-1, -1])

    # get 15-arcsecond SRTM height data for the ABI region
    srtm_lat, srtm_lon, srtm_height = get_ellipsoidal_srtm(
        srtm_nc_path=srtm_nc_path,
        lat_bounds=abi_lat_bounds,
        lon_bounds=abi_lon_bounds,
    )
    srtm_lat_bounds = srtm_lat[0, 0], srtm_lat[-1, -1]
    srtm_lon_bounds = srtm_lon[0, 0], srtm_lon[-1, -1]

    # SRTM15 includes bathymetry, so only look down to -10 m
    srtm_height = np.where(srtm_height >= -10, srtm_height, 0)

    # warp SRTM heights to the projection of the ABI scene
    warped_heights = original_abi_img.resample2abi(
        srtm_height,
        lat_bounds=srtm_lat_bounds,
        lon_bounds=srtm_lon_bounds,
        resample_algo="cubicspline",
    )

    # make new images using ABI Fixed Grid lat/lon and the warped SRTM heights

    # if resample_nav=False (default), the ABI image is orthorectified to correct for terrain parallax
    img_resampled_to_nav = ABIImage(
        abi_nc_path,
        gamma=abi_gamma,
        lat_bounds=original_abi_img.lat_deg,
        lon_bounds=original_abi_img.lon_deg,
        height_m=warped_heights,
        resample_nav=False,
    )

    # if resample_nav=True, the underlying navigation is orthorectified and image pixels are untouched
    nav_resampled_to_img = ABIImage(
        abi_nc_path,
        gamma=abi_gamma,
        lat_bounds=original_abi_img.lat_deg,
        lon_bounds=original_abi_img.lon_deg,
        height_m=warped_heights,
        resample_nav=True,
    )

    # save images with our target coordinates indicated in green
    save_path = Path("img")
    save_path.mkdir(exist_ok=True)
    cv2.imwrite(
        save_path.joinpath("original.png"),
        abi_latlon_target(original_abi_img, target_latlon),
    )
    cv2.imwrite(
        save_path.joinpath("resampled-image.png"),
        abi_latlon_target(img_resampled_to_nav, target_latlon),
    )
    cv2.imwrite(
        save_path.joinpath("resampled-nav.png"),
        abi_latlon_target(nav_resampled_to_img, target_latlon),
    )


def abi_latlon_target(
    img: ABIImage,
    target_latlon: tuple[float, float],
    color_bgr: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    upscale_factor = 2
    upscaled_bv = nearest_scale(img.bv, upscale_factor)

    marked_img = np.stack((upscaled_bv,) * 3, axis=-1).astype(np.uint8)
    for target_lat, target_lon in target_latlon:
        target_idx = nearest_2d_search(
            y_arr=img.lat_deg,
            x_arr=img.lon_deg,
            target_y=np.atleast_1d(target_lat),
            target_x=np.atleast_1d(target_lon),
        )

        target_idx = scale_idx(target_idx, upscale_factor)

        marked_img[*target_idx, :] = color_bgr
        cv2.circle(marked_img, center=target_idx[::-1], radius=15, color=color_bgr)

    return marked_img


if __name__ == "__main__":
    demo()
