# Copyright (c) 2023-2025.

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

import os
from pathlib import Path

import netCDF4
import numpy as np


class _NCBase:
    def __init__(self):
        pass

    def _copy_attrs(self, src_obj, dst_obj, attr_list):
        for attr_name in attr_list:
            attr_value = getattr(src_obj, attr_name)
            if not callable(attr_value):
                setattr(dst_obj, attr_name, attr_value)


class _NCVarBase:
    def __init__(self):
        pass

    def __getitem__(self, key):
        return getattr(self, key)


class NCInterface(_NCBase):
    """
    Walkable interface for netCDF4 Dataset

    Access netCDF4 variables under `.variables`, and dimensions under `.dimensions`.
    Masked variables are always filled, and scalar variables are always 1D arrays.

    `nci = NCInterface("my_netcdf.nc")`
    `nci.variables.MyVariable[...]` #, or:
    `nci["MyVariable"][...]`

    If a variable in the netCDF is masked, it gets filled and returned as an ndarray.
    Override the _FillValue for a variable:

    `nci.variables.MyVariable.set_fill_value(np.nan)`

    If the size of the variable array is >= `lazy_load_size_threshold`,
    the array is not loaded from disk until it is accessed with an Ellipsis or slice:

    `nci.variables.MyVariable[...]`
    `nci.["MyVariable"][0,0]`
    `nci.variables.MyVariable[np.s_[0:100, 0:100]]`

    *Note: While the shape of the array can change when indexed, the `dimensions` attribute of the variable does not.*
    """

    def __init__(
        self, nc_file: str | os.PathLike, lazy_load_size_threshold: int = 500 * 500
    ):
        self._loaded_nc = None
        self._nc_file = Path(nc_file)

        self._loaded_nc = netCDF4.Dataset(self._nc_file, "r")

        # set attributes
        self._copy_attrs(
            src_obj=self._loaded_nc, dst_obj=self, attr_list=self._loaded_nc.ncattrs()
        )

        # set dimensions
        self.dimensions = {}
        for dimension_name, dimension_obj in self._loaded_nc.dimensions.items():
            self.dimensions[dimension_name] = _NCDim(dimension_name)
            setattr(self.dimensions[dimension_name], "size", dimension_obj.size)

        # set variables
        self.variables = _NCVarBase()

        for var_name in self._loaded_nc.variables.keys():
            var_obj = self._loaded_nc.variables[var_name]

            # set this variable to be lazy-loading only if it's ge lazy_load_size_threshold
            if var_obj.size >= lazy_load_size_threshold:
                lazy_load = True

            else:
                lazy_load = False

            setattr(
                self.variables, var_name, _NCVar(self._loaded_nc, var_name, lazy_load)
            )

            var_attrs = var_obj.ncattrs()
            var_attrs.extend(getattr(getattr(self.variables, var_name), "__npattrs__"))
            self._copy_attrs(
                src_obj=var_obj,
                dst_obj=getattr(self.variables, var_name),
                attr_list=var_attrs,
            )
            setattr(getattr(self.variables, var_name), "__ncattrs__", var_attrs)

            # add dimension reference to the variable
            var_dims = []
            for var_dim in var_obj.dimensions:
                var_dims.append(self.dimensions[var_dim])
            setattr(getattr(self.variables, var_name), "dimensions", tuple(var_dims))

    def __getitem__(self, key):
        return getattr(self.variables, key)

    def __del__(self):
        if self._loaded_nc is not None:
            self._loaded_nc.close()


class _NCVar(_NCBase):
    def __init__(self, nc_obj, var_name, lazy_load):
        self._loaded_nc = nc_obj

        self.name = var_name

        self.__ncattrs__ = [None]
        self.__npattrs__ = ["dtype", "ndim", "shape", "size"]

        self.__value__ = None
        self._mask = None
        self._is_loaded = False
        self._slc = np.s_[...]
        self._override_fill_value = None

        if not lazy_load:
            self._loadvar()

    @property
    def mask(self):
        if not self._is_loaded:
            self._loadvar()

        return self._mask

    def set_fill_value(self, val):
        if val != self._FillValue:
            self._override_fill_value = val
            self._FillValue = val

            # reload with the new fill value
            if self._is_loaded:
                self._loadvar()

    def _loadvar(self):
        var = self._loaded_nc.variables[self.name]

        if self._slc is Ellipsis or isinstance(self._slc, (int, slice)):
            self.__value__ = var[self._slc]
        else:
            # Normalize to tuple
            slc_tuple = (self._slc,) if isinstance(self._slc, np.ndarray) else self._slc

            inner_slc = []
            outer_slc = []

            for s in slc_tuple:
                arr_s = np.asarray(s)
                if arr_s.ndim > 1:
                    inner_slc.append(slice(arr_s.min(), arr_s.max() + 1))
                    outer_slc.append(arr_s - arr_s.min())

            if inner_slc:  # if any multidimensional indices found
                self.__value__ = var[tuple(inner_slc)][tuple(outer_slc)]
            else:
                self.__value__ = var[self._slc]

        # handle mask and fill
        if not np.ma.isMaskedArray(self.__value__):
            self.__value__ = np.atleast_1d(self.__value__)

        else:
            self._mask = self.__value__.mask

            fill_value = None
            if self._override_fill_value is not None:
                fill_value = self._override_fill_value

            elif hasattr(var, "_FillValue"):
                fill_value = var._FillValue

            self.__value__ = np.ma.masked_array(
                self.__value__, mask=self._mask, fill_value=fill_value
            )
            self.pct_unmasked = self.__value__.count() / max(self.__value__.size, 1)
            self.__value__ = self.__value__.filled()

        # gather updated attrs of the numpy array
        super()._copy_attrs(self.__value__, self, self.__npattrs__)

        self._is_loaded = True

    def _slice_changed(self, slc):
        # return true if the slice changed
        slc = np.s_[slc]

        return not (np.array_equal(self._slc, slc))

    def _slice_var(self, slc):
        slc = np.s_[slc]
        if not self._is_loaded or self._slice_changed(slc):
            self._slc = slc
            self._loadvar()

    def __getitem__(self, slc):
        self._slice_var(slc)

        return self.__value__

    def __setitem__(self, slc, val):
        self._slice_var(slc)

        self.__value__[...] = val

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        retval = []

        retval.append(repr(type(self)))

        ncdims = []
        for ncdim in self.dimensions:
            ncdims.append(ncdim.name)

        retval.append(
            str(self.dtype) + f" {self.name}(" + ", ".join(tuple(ncdims)) + ")"
        )

        for attr in sorted(self.__ncattrs__):
            retval.append(f"    {attr}: " + str(getattr(self, attr)))

        return "\n".join(retval)


class _NCDim(_NCBase):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "%r: name = '%s', size = %s" % (type(self), self.name, self.size)

    def __getitem__(self, key):
        return getattr(self, key)
