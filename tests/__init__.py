# Copyright (c) 2022-2023.

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

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_PATH.parent.resolve()))

import heregoes

input_dir = SCRIPT_PATH.joinpath("input")
input_dir.mkdir(parents=True, exist_ok=True)

output_dir = SCRIPT_PATH.joinpath("output")
output_dir.mkdir(parents=True, exist_ok=True)

for output_file in output_dir.glob("*"):
    output_file.unlink()
