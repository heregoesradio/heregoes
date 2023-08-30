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

import logging
import time

logger = logging.getLogger("heregoes-logger")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03dZ | %(levelname)s | %(caller)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    style="%",
)
formatter.converter = time.gmtime
handler.setFormatter(formatter)
logger.addHandler(handler)


class HereGOESException(Exception):
    def __init__(self, caller, msg, exception=None):
        if exception is not None:
            msg += f" Exception: {exception}"
        logger.critical(msg, extra={"caller": caller})
        super().__init__(msg)


class HereGOESIOError(HereGOESException):
    pass


class HereGOESIOReadException(HereGOESIOError):
    def __init__(self, caller, filepath, exception=None):
        super().__init__(
            caller=caller,
            msg=f"Could not read from file at {filepath}.",
            exception=exception,
        )


class HereGOESIOWriteException(HereGOESIOError):
    def __init__(self, caller, filepath, exception=None):
        super().__init__(
            caller=caller,
            msg=f"Could not write to file at {filepath}.",
            exception=exception,
        )


class HereGOESValueError(HereGOESException):
    pass


class HereGOESUnsupportedProductException(HereGOESValueError):
    def __init__(self, caller, filepath, exception=None):
        super().__init__(
            caller=caller,
            msg=f"Product type is not supported: {filepath}.",
            exception=exception,
        )
