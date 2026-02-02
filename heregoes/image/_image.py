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

import re
from pathlib import Path

import cv2


class _Image:
    """
    An image is an array of 8-bit brightness values that can be saved to JPEG or PNG with cv2
    """

    def save(self, filepath=Path("."), ext=".png", source="bv"):
        filepath = Path(filepath)

        # if a directory path is provided instead of a file path, append a default filename
        if filepath.is_dir():
            filepath = filepath.joinpath(self.default_filename)

        # append an image suffix if not in the file path
        if not (filepath.suffix):
            filepath = filepath.with_suffix(ext)

        file_dir = filepath.parent.resolve()
        file_dir.mkdir(parents=True, exist_ok=True)

        if bool(re.search(r"\.jp[e]?g", filepath.suffix.lower())):
            cv2_quality = [cv2.IMWRITE_JPEG_QUALITY, 100]

        elif bool(re.search(r"\.png", filepath.suffix.lower())):
            cv2_quality = [cv2.IMWRITE_PNG_COMPRESSION, 9]

        else:
            cv2_quality = None

        try:
            result = cv2.imwrite(str(filepath), getattr(self, source), cv2_quality)
            if not result:
                raise Exception("cv2.imwrite returned False.")

        except Exception as e:
            raise IOError(f"Failed to write image at {filepath}. Exception: {e}")

        else:
            return filepath
