# Copyright (c) 2023.

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

"""Partial implementation of a flat netCDF4 Python class interface with lazy-loading netCDF variables"""

from pathlib import Path

import netCDF4
import numpy as np

from heregoes import exceptions


class _NCBase:
    # empty class just for interface structure
    pass


class NCInterface(_NCBase):
    def __init__(self, nc_file, lazy_load_size_threshold=500**2):
        self._loaded_nc = None

        try:
            self._nc_file = Path(nc_file)

            if not (
                self._nc_file.exists()
                and (self._nc_file.is_file() or self._nc_file.is_symlink())
            ):
                raise exceptions.HereGOESIOError

        except Exception as e:
            raise exceptions.HereGOESIOReadException(
                caller=f"{__name__}.{self.__class__.__name__}",
                filepath=self._nc_file,
                exception=e,
            )

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
        self.variables = _NCBase()

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
            var_attrs.extend(["ndim", "shape", "size", "dtype"])
            self._copy_attrs(
                src_obj=var_obj,
                dst_obj=getattr(self.variables, var_name),
                attr_list=var_attrs,
            )

            # add dimension reference to the variable
            var_dims = []
            for var_dim in var_obj.dimensions:
                var_dims.append(self.dimensions[var_dim])
            setattr(getattr(self.variables, var_name), "dimensions", tuple(var_dims))

    def _copy_attrs(self, src_obj, dst_obj, attr_list):
        for attr_name in attr_list:
            if not (attr_name.startswith("_")):
                attr_value = getattr(src_obj, attr_name)
                if not (callable(attr_value)):
                    setattr(dst_obj, attr_name, attr_value)

    def __getitem__(self, key):
        return getattr(self.variables, key)

    def __del__(self):
        if self._loaded_nc is not None:
            self._loaded_nc.close()


class _NCVar(_NCBase):
    def __init__(self, nc_obj, var_name, lazy_load):
        self._loaded_nc = nc_obj

        self.var_name = var_name

        self.__value = None
        self._override_fill_value = None

        if not lazy_load:
            self._loadvar()

    def _loadvar(self):
        self.__value = self._loaded_nc.variables[self.var_name][...]

        self.quality = self.__value.count() / self.__value.size

        self.__value = np.atleast_1d(self.__value)

        if self._override_fill_value is not None:
            self.__value.set_fill_value(self._override_fill_value)

        self.__value = self.__value.filled()

    @property
    def _value(self):
        if self.__value is None:
            self._loadvar()

        return self.__value

    @_value.setter
    def _value(self, val):
        self.__value = val

    def set_fill_value(self, val):
        self._override_fill_value = val
        self._loadvar()

    def __array__(self):
        return self.__getitem__(...)

    def __getitem__(self, key):
        return self._value[key]

    def __setitem__(self, key, val):
        self._value[key] = val

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        retval = []

        retval.append(repr(type(self)))

        ncdims = []
        for ncdim in self.dimensions:
            ncdims.append(ncdim.name)

        retval.append(
            str(self.dtype) + f" {self.var_name}(" + ", ".join(tuple(ncdims)) + ")"
        )

        for attr in self.__dict__.keys():
            if not attr.startswith("_"):
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
