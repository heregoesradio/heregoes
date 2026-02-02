import numpy as np
import pytest

from heregoes import load
from heregoes.util import nearest_2d_search
from tests.resources_20190904 import abi_fc02_nc


def test_noisy_nearest_2d():
    abi_data = load(abi_fc02_nc)

    center_idx = 10848

    # search in this area
    search_pad = 1000
    search_slc = np.s_[center_idx - search_pad : center_idx + search_pad]

    # for this target range
    target_pad = 20
    target_slc = np.s_[center_idx - target_pad : center_idx + target_pad]

    search_x_mesh, search_y_mesh = np.meshgrid(
        abi_data.variables.x[search_slc], abi_data.variables.y[search_slc]
    )
    target_x_mesh, target_y_mesh = np.meshgrid(
        abi_data.variables.x[target_slc], abi_data.variables.y[target_slc]
    )

    noise_pct = 0.49
    target_y_noise = np.random.uniform(
        0, noise_pct * abi_data.resolution_ifov, target_y_mesh.shape
    )
    target_x_noise = np.random.uniform(
        0, noise_pct * abi_data.resolution_ifov, target_x_mesh.shape
    )

    # add random noise to the targets to test nearest neighbor
    noisy_target_y_mesh = target_y_mesh + target_y_noise
    noisy_target_x_mesh = target_x_mesh + target_x_noise

    search_2d_in_1d = nearest_2d_search(
        y_arr=search_y_mesh[:, 0],
        x_arr=search_x_mesh[0, :],
        target_y=noisy_target_y_mesh,
        target_x=noisy_target_x_mesh,
    )
    assert (target_y_mesh == search_y_mesh[search_2d_in_1d]).all()
    assert (target_x_mesh == search_x_mesh[search_2d_in_1d]).all()

    search_1d_in_2d = nearest_2d_search(
        y_arr=search_y_mesh,
        x_arr=search_x_mesh,
        target_y=noisy_target_y_mesh[:, 0],
        target_x=noisy_target_x_mesh[0, :],
    )
    assert (target_y_mesh == search_y_mesh[search_1d_in_2d]).all()
    assert (target_x_mesh == search_x_mesh[search_1d_in_2d]).all()

    search_2d_in_2d = nearest_2d_search(
        y_arr=search_y_mesh,
        x_arr=search_x_mesh,
        target_y=noisy_target_y_mesh,
        target_x=noisy_target_x_mesh,
    )
    assert (target_y_mesh == search_y_mesh[search_2d_in_2d]).all()
    assert (target_x_mesh == search_x_mesh[search_2d_in_2d]).all()

    search_1d_in_1d = nearest_2d_search(
        y_arr=search_y_mesh[:, 0],
        x_arr=search_x_mesh[0, :],
        target_y=noisy_target_y_mesh[:, 0],
        target_x=noisy_target_x_mesh[0, :],
    )
    assert (target_y_mesh == search_y_mesh[search_1d_in_1d]).all()
    assert (target_x_mesh == search_x_mesh[search_1d_in_1d]).all()
