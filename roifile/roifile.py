# roifile.py

# Copyright (c) 2020-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read and write ImageJ ROI format.

Roifile is a Python library to read, write, create, and plot `ImageJ`_ ROIs,
an undocumented and ImageJ application specific format to store regions of
interest, geometric shapes, paths, text, and whatnot for image overlays.

.. _ImageJ: https://imagej.net

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2026.2.10
:DOI: `10.5281/zenodo.6941603 <https://doi.org/10.5281/zenodo.6941603>`_

Quickstart
----------

Install the roifile package and all dependencies from the
`Python Package Index <https://pypi.org/project/roifile/>`_::

    python -m pip install -U "roifile[all]"

View overlays stored in a ROI, ZIP, or TIFF file::

    python -m roifile file.roi

See `Examples`_ for using the programming interface.

Source code, examples, and support are available on
`GitHub <https://github.com/cgohlke/roifile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.11.9, 3.12.10, 3.13.12, 3.14.3 64-bit
- `NumPy <https://pypi.org/project/numpy>`_ 2.4.2
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2026.1.28 (optional)
- `Imagecodecs <https://pypi.org/project/imagecodecs/>`_ 2026.1.14 (optional)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.10.8 (optional)

Revisions
---------

2026.2.10

- Revise wrapping of integer coordinates again (breaking).
- Bump file version to 229.
- Support groups > 255 (untested).
- Support IMAGE subtype (requires imagecodecs).
- Add point_type and point_size properties for point ROIs.
- Do not return empty paths in path2coords.
- Improve documentation.

2026.1.29

- Fix code review issues.

2026.1.22

- Fix boolean codec in ImagejRoi.properties.

2026.1.20

- Fix reading ImagejRoi.props.
- Add ImagejRoi.properties property to decode and encode ImagejRoi.props.

2026.1.8

- Improve code quality.
- Drop support for Python 3.10.

2025.12.12

- Move tests to separate module.

2025.5.10

- Support Python 3.14.

2025.2.20

- …

Refer to the CHANGES file for older revisions.

Notes
-----

The ImageJ ROI format cannot store integer coordinate values outside the
range of -5000..60536.

Refer to the ImageJ `RoiDecoder.java
<https://github.com/imagej/ImageJ/blob/master/ij/io/RoiDecoder.java>`_
source code for a reference implementation.

Other Python packages handling ImageJ ROIs:

- `ijpython_roi <https://github.com/dwaithe/ijpython_roi>`_
- `read-roi <https://github.com/hadim/read-roi/>`_
- `sdt-python <https://github.com/schuetzgroup/sdt-python>`_
- `napari_jroitools <https://github.com/jayunruh/napari_jroitools>`_

Examples
--------

Create a new ImagejRoi instance from an array of x, y coordinates,
then set ROI properties:

>>> roi = ImagejRoi.frompoints([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
>>> roi.roitype = ROI_TYPE.POINT
>>> roi.point_size = ROI_POINT_SIZE.LARGE
>>> roi.options |= ROI_OPTIONS.SHOW_LABELS

Export the instance to an ImageJ ROI formatted byte string or file:

>>> out = roi.tobytes()
>>> out[:4]
b'Iout'
>>> roi.tofile('_test.roi')

Read the ImageJ ROI from the file and verify the content:

>>> roi2 = ImagejRoi.fromfile('_test.roi')
>>> roi2 == roi
True
>>> roi.roitype == ROI_TYPE.POINT
True
>>> roi.subpixelresolution
True
>>> roi.coordinates()
array([[1.1, 2.2],
       [3.3, 4.4],
       [5.5, 6.6]], dtype=float32)
>>> roi.left, roi.top, roi.right, roi.bottom
(1, 2, 7, 8)
>>> roi2.name = 'test'

Plot the ROI using matplotlib:

>>> roi.plot()

Write the ROIs to a ZIP file:

>>> roiwrite('_test.zip', [roi, roi2], mode='w')

Read the ROIs from the ZIP file:

>>> rois = roiread('_test.zip')
>>> assert len(rois) == 2 and rois[0] == roi and rois[1].name == 'test'

Write the ROIs to an ImageJ formatted TIFF file:

>>> import numpy
>>> import tifffile
>>> tifffile.imwrite(
...     '_test.tif',
...     numpy.zeros((9, 9), 'u1'),
...     imagej=True,
...     metadata={'Overlays': [roi.tobytes(), roi2.tobytes()]},
... )

Read the ROIs embedded in an ImageJ formatted TIFF file:

>>> rois = roiread('_test.tif')
>>> assert len(rois) == 2 and rois[0] == roi and rois[1].name == 'test'

View the overlays stored in a ROI, ZIP, or TIFF file from a command line::

    python -m roifile _test.roi

For an advanced example, see `roifile_demo.py` in the source distribution.

"""

from __future__ import annotations

__version__ = '2026.2.10'

__all__ = [
    'ROI_COLOR_NONE',
    'ROI_OPTIONS',
    'ROI_POINT_SIZE',
    'ROI_POINT_TYPE',
    'ROI_SUBTYPE',
    'ROI_TYPE',
    'ImagejRoi',
    '__version__',
    'logger',
    'roiread',
    'roiwrite',
]

import contextlib
import enum
import logging
import os
import struct
import sys
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Literal

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray


def roiread(
    filename: os.PathLike[Any] | str,
    /,
    *,
    min_int_coord: int | None = None,
    maxsize: int = 268435456,  # 256 MB
) -> ImagejRoi | list[ImagejRoi]:
    """Return ImagejRoi instance(s) from ROI, ZIP, or TIFF file.

    For ZIP or TIFF files, return a list of ImagejRoi.

    Parameters:
        filename: Path to ROI, ZIP, or TIFF file.
        min_int_coord: Minimum integer coordinate value for unwrapping.
            Default is -5000 (ImageJ standard).
        maxsize: Maximum file size to read in bytes.

    Returns:
        Single ImagejRoi instance for .roi files,
        or list of ImagejRoi instances for .zip and .tif files.

    """
    return ImagejRoi.fromfile(
        filename, min_int_coord=min_int_coord, maxsize=maxsize
    )


def roiwrite(
    filename: os.PathLike[Any] | str,
    roi: ImagejRoi | Iterable[ImagejRoi],
    /,
    *,
    name: str | Iterable[str] | None = None,
    mode: Literal['w', 'x', 'a'] | None = None,
) -> None:
    """Write ImagejRoi instance(s) to ROI or ZIP file.

    Write an ImagejRoi instance to a ROI file or write a sequence of ImagejRoi
    instances to a ZIP file. Existing ZIP files are opened for append.

    Parameters:
        filename: Path to output .roi or .zip file.
        roi: Single ImagejRoi instance or iterable of instances.
        name: Optional name(s) for ROI(s) in ZIP file.
        mode: File mode ('w' for write, 'x' for exclusive, 'a' for append).
            Defaults to 'a' for existing files, 'w' for new files.

    """
    filename = os.fspath(filename)

    if isinstance(roi, ImagejRoi):
        assert name is None or isinstance(name, str)
        return roi.tofile(filename, name=name, mode=mode)

    if mode is None:
        mode = 'a' if os.path.exists(filename) else 'w'

    if name is not None:
        if isinstance(name, str):
            msg = "'name' is not an iterable of str"
            raise ValueError(msg)
        name = iter(name)

    import zipfile

    assert mode is not None
    with zipfile.ZipFile(filename, mode) as zf:
        for r in roi:
            if name is None:
                n = r.name or r.autoname
            else:
                try:
                    n = next(name)
                except StopIteration:
                    msg = "'name' iterator has fewer items than 'roi'"
                    raise ValueError(msg) from None
            if not n:
                n = r.autoname
            n = n if n[-4:].lower() == '.roi' else n + '.roi'
            with zf.open(n, 'w') as fh:
                fh.write(r.tobytes())
    return None


class ROI_TYPE(enum.IntEnum):
    """ImageJ ROI types."""

    UNKNOWN = -1
    """Undocumented or unknown ROI type."""
    POLYGON = 0
    """Polygon with straight edges."""
    RECT = 1
    """Rectangle."""
    OVAL = 2
    """Oval/ellipse."""
    LINE = 3
    """Straight line."""
    FREELINE = 4
    """Freehand line."""
    POLYLINE = 5
    """Polyline with straight segments."""
    NOROI = 6
    """No ROI."""
    FREEHAND = 7
    """Freehand polygon."""
    TRACED = 8
    """Traced polygon."""
    ANGLE = 9
    """Angle measurement."""
    POINT = 10
    """Point or multi-point."""

    @classmethod
    def _missing_(cls, value: object) -> Self:
        assert isinstance(value, int)
        obj = cls(-1)
        obj._value_ = value
        return obj


class ROI_SUBTYPE(enum.IntEnum):
    """ImageJ ROI subtypes."""

    UNKNOWN = -1
    """Undocumented or unknown ROI subtype."""
    UNDEFINED = 0
    """No subtype specified."""
    TEXT = 1
    """Text overlay."""
    ARROW = 2
    """Arrow overlay."""
    ELLIPSE = 3
    """Ellipse (fitted)."""
    IMAGE = 4
    """Embedded image overlay."""
    ROTATED_RECT = 5
    """Rotated rectangle."""

    @classmethod
    def _missing_(cls, value: object) -> Self:
        assert isinstance(value, int)
        obj = cls(-1)
        obj._value_ = value
        return obj


class ROI_OPTIONS(enum.IntFlag):
    """ImageJ ROI options."""

    NONE = 0
    """No options."""
    SPLINE_FIT = 1
    """Spline fit enabled."""
    DOUBLE_HEADED = 2
    """Double-headed arrow."""
    OUTLINE = 4
    """Outline only (no fill)."""
    OVERLAY_LABELS = 8
    """Show overlay labels."""
    OVERLAY_NAMES = 16
    """Show overlay names."""
    OVERLAY_BACKGROUNDS = 32
    """Show overlay backgrounds."""
    OVERLAY_BOLD = 64
    """Bold overlay text."""
    SUB_PIXEL_RESOLUTION = 128
    """Subpixel resolution coordinates."""
    DRAW_OFFSET = 256
    """Draw with offset."""
    ZERO_TRANSPARENT = 512
    """Zero values transparent."""
    SHOW_LABELS = 1024
    """Show point labels."""
    SCALE_LABELS = 2048
    """Scale labels with zoom."""
    PROMPT_BEFORE_DELETING = 4096
    """Prompt before deletion."""
    SCALE_STROKE_WIDTH = 8192
    """Scale stroke width with zoom."""
    UNKNOWN_14 = 16384
    """Undocumented or unknown option (bit 14)."""
    UNKNOWN_15 = 32768
    """Undocumented or unknown option (bit 15)."""


class ROI_POINT_TYPE(enum.IntEnum):
    """ImageJ ROI point types."""

    UNKNOWN = -1
    """Undocumented or unknown point type."""
    HYBRID = 0
    """Hybrid marker (cross with dot)."""
    CROSS = 1
    """Cross/crosshair marker."""
    # CROSSHAIR = 1  # alias for CROSS in ImageJ, not needed here
    DOT = 2
    """Dot/circle marker (filled)."""
    CIRCLE = 3
    """Circle marker (outline)."""

    @classmethod
    def _missing_(cls, value: object) -> Self:
        assert isinstance(value, int)
        obj = cls(-1)
        obj._value_ = value
        return obj


class ROI_POINT_SIZE(enum.IntEnum):
    """ImageJ ROI point sizes."""

    UNKNOWN = -1
    """Undocumented or unknown point size."""
    TINY = 1
    """Tiny marker (1px)."""
    SMALL = 3
    """Small marker (3px)."""
    MEDIUM = 5
    """Medium marker (5px)."""
    LARGE = 7
    """Large marker (7px)."""
    EXTRA_LARGE = 11
    """Extra large marker (11px)."""
    XXL = 17
    """XXL marker (17px)."""
    XXXL = 25
    """XXXL marker (25px)."""

    @classmethod
    def _missing_(cls, value: object) -> Self:
        assert isinstance(value, int)
        obj = cls(-1)
        obj._value_ = value
        return obj


ROI_COLOR_NONE = b'\x00\x00\x00\x00'
"""No color or black."""


@dataclass
class ImagejRoi:
    """Read and write ImageJ ROI format."""

    byteorder: Literal['>', '<'] = '>'
    """Byte order: '>' for big-endian, '<' for little-endian."""
    roitype: ROI_TYPE = ROI_TYPE.POLYGON
    """ROI type (polygon, rect, oval, line, point, etc)."""
    subtype: ROI_SUBTYPE = ROI_SUBTYPE.UNDEFINED
    """Subtype for specialized ROIs (text, arrow, ellipse, image, etc)."""
    options: ROI_OPTIONS = ROI_OPTIONS.NONE
    """Bit flags for ROI options and features."""
    name: str = ''
    """ROI name."""
    props: str = ''
    """Properties string containing key:value pairs."""
    version: int = 229
    """File format version number."""
    top: int = 0
    """Bounding rectangle top coordinate."""
    left: int = 0
    """Bounding rectangle left coordinate."""
    bottom: int = 0
    """Bounding rectangle bottom coordinate."""
    right: int = 0
    """Bounding rectangle right coordinate."""
    n_coordinates: int = 0
    """Number of coordinate pairs."""
    stroke_width: int = 0
    """Stroke width in pixels (also point_size for POINT ROIs version 226+)."""
    shape_roi_size: int = 0
    """Composite shape data size in floats."""
    stroke_color: bytes = ROI_COLOR_NONE
    """Stroke/outline color as ARGB bytes."""
    fill_color: bytes = ROI_COLOR_NONE
    """Fill color as ARGB bytes."""
    arrow_style_or_aspect_ratio: int = 0
    """Arrow style, aspect ratio for ellipse, or point_type for POINT ROIs."""
    arrow_head_size: int = 0
    """Arrow head size in pixels."""
    rounded_rect_arc_size: int = 0
    """Arc size for rounded rectangle corners."""
    position: int = 0
    """Position in stack (1-based, 0 means not set)."""
    c_position: int = 0
    """Channel position (1-based, 0 means not set)."""
    z_position: int = 0
    """Z-slice position (1-based, 0 means not set)."""
    t_position: int = 0
    """Time frame position (1-based, 0 means not set)."""
    x1: float = 0.0
    """First X coordinate for line ROIs or X for subpixel rectangles."""
    y1: float = 0.0
    """First Y coordinate for line ROIs or Y for subpixel rectangles."""
    x2: float = 0.0
    """Second X coordinate for line ROIs."""
    y2: float = 0.0
    """Second Y coordinate for line ROIs."""
    xd: float = 0.0
    """X coordinate for subpixel rectangles (double precision)."""
    yd: float = 0.0
    """Y coordinate for subpixel rectangles (double precision)."""
    widthd: float = 0.0
    """Width for subpixel rectangles (double precision)."""
    heightd: float = 0.0
    """Height for subpixel rectangles (double precision)."""
    overlay_label_color: bytes = ROI_COLOR_NONE
    """Overlay label color as ARGB bytes."""
    overlay_font_size: int = 0
    """Overlay label font size in points."""
    group: int = 0
    """Group number for grouping related ROIs."""
    image_opacity: int = 0
    """Opacity for image ROIs (0-255)."""
    image_size: int = 0
    """Embedded image data size in bytes."""
    image_data: bytes | None = None
    """Embedded image data for IMAGE subtype ROIs."""
    float_stroke_width: float = 0.0
    """Floating point stroke width for precise rendering."""
    text_size: int = 0
    """Text ROI font size in points."""
    text_style: int = 0
    """Text ROI font style flags (bold, italic, etc)."""
    text_justification: int = 0
    """Text ROI alignment (left, center, right)."""
    text_angle: float = 0.0
    """Text ROI rotation angle in degrees."""
    text_name: str = ''
    """Text ROI font name."""
    text: str = ''
    """Text ROI content."""
    counters: NDArray[numpy.uint8] | None = None
    """Counter values for each coordinate point."""
    counter_positions: NDArray[numpy.uint32] | None = None
    """Counter positions for each coordinate point."""
    integer_coordinates: NDArray[numpy.int32] | None = None
    """Integer coordinate pairs relative to bounding box."""
    subpixel_coordinates: NDArray[numpy.float32] | None = None
    """Floating point coordinate pairs for subpixel precision."""
    multi_coordinates: NDArray[numpy.float32] | None = None
    """Path data for composite shapes (MOVETO, LINETO, CLOSE operations)."""

    @classmethod
    def frompoints(
        cls,
        points: ArrayLike | None = None,
        /,
        *,
        name: str | None = None,
        position: int | None = None,
        index: int | str | None = None,
        c: int | None = None,
        z: int | None = None,
        t: int | None = None,
    ) -> Self:
        """Return ImagejRoi instance from sequence of Point coordinates.

        Use floating point coordinates for subpixel precision or values outside
        the range -5000..60535.

        A FREEHAND ROI with options OVERLAY_BACKGROUNDS and OVERLAY_LABELS is
        returned.

        Parameters:
            points: Array-like of shape (n, 2) containing x, y coordinates.
            name: ROI name. Auto-generated if None.
            position: Stack position (0-based). Stored as 1-based.
            index: Index for auto-generated name.
            c: Channel position (0-based). Stored as 1-based.
            z: Z-slice position (0-based). Stored as 1-based.
            t: Time frame position (0-based). Stored as 1-based.

        Returns:
            New ImagejRoi instance created from points.

        """
        if points is None:
            return cls()

        self = cls()
        self.roitype = ROI_TYPE.FREEHAND
        self.options = (
            ROI_OPTIONS.OVERLAY_BACKGROUNDS | ROI_OPTIONS.OVERLAY_LABELS
        )
        if position is not None:
            self.position = position + 1
        if c is not None:
            self.c_position = c + 1
        if z is not None:
            self.z_position = z + 1
        if t is not None:
            self.t_position = t + 1
        if name is None:
            if index is None:
                name = str(uuid.uuid1())
            else:
                name = f'F{self.t_position:02}-C{index}'
        self.name = name

        coords = numpy.array(points, copy=True)
        if coords.size == 0:
            msg = 'points array is empty'
            raise ValueError(msg)
        if coords.ndim != 2 or coords.shape[1] != 2:
            msg = f'invalid points array shape {coords.shape}, expected (n, 2)'
            raise ValueError(msg)
        if coords.dtype.kind == 'f' or (
            numpy.any(coords > 60535) or numpy.any(coords < -5000)
        ):
            self.options |= ROI_OPTIONS.SUB_PIXEL_RESOLUTION
            self.subpixel_coordinates = coords.astype(numpy.float32, copy=True)
            if coords.dtype.kind == 'f':
                coords = numpy.round(coords)

        coords = numpy.array(coords, dtype=numpy.int32)

        left_top = coords.min(axis=0)
        right_bottom = coords.max(axis=0)
        right_bottom += [1, 1]

        coords -= left_top
        self.integer_coordinates = coords
        self.n_coordinates = len(self.integer_coordinates)
        self.left = int(left_top[0])
        self.top = int(left_top[1])
        self.right = int(right_bottom[0])
        self.bottom = int(right_bottom[1])

        return self

    @classmethod
    def fromfile(
        cls,
        filename: os.PathLike[Any] | str,
        /,
        *,
        min_int_coord: int | None = None,
        maxsize: int = 268435456,  # 256 MB
    ) -> ImagejRoi | list[ImagejRoi]:
        """Return ImagejRoi instance from ROI, ZIP, or TIFF file.

        For ZIP or TIFF files, return a list of ImagejRoi.

        Parameters:
            filename: Path to .roi, .zip, or .tif file.
            min_int_coord: Minimum integer coordinate for unwrapping.
            maxsize: Maximum bytes to read per entry.

        Returns:
            Single ImagejRoi for .roi files, list of ImagejRoi for .zip/.tif.

        """
        filename = os.fspath(filename)
        if filename[-4:].lower() == '.tif':
            import tifffile

            with tifffile.TiffFile(filename) as tif:
                if tif.imagej_metadata is None:
                    msg = 'file does not contain ImagejRoi'
                    raise ValueError(msg)
                rois: list[bytes] = []
                if 'Overlays' in tif.imagej_metadata:
                    overlays = tif.imagej_metadata['Overlays']
                    if isinstance(overlays, (list, tuple)):
                        rois.extend(overlays)
                    else:
                        rois.append(overlays)
                if 'ROI' in tif.imagej_metadata:
                    roi = tif.imagej_metadata['ROI']
                    if isinstance(roi, (list, tuple)):
                        rois.extend(roi)
                    else:
                        rois.append(roi)
                return [
                    cls.frombytes(roi, min_int_coord=min_int_coord)
                    for roi in rois
                ]

        if filename[-4:].lower() == '.zip':
            import zipfile

            with zipfile.ZipFile(filename) as zf:
                return [
                    cls.frombytes(
                        zf.open(name).read(maxsize),
                        min_int_coord=min_int_coord,
                    )
                    for name in zf.namelist()
                ]

        with open(filename, 'rb') as fh:
            data = fh.read(maxsize)
        return cls.frombytes(data, min_int_coord=min_int_coord)

    @classmethod
    def frombytes(
        cls,
        data: bytes,
        /,
        *,
        min_int_coord: int | None = None,
    ) -> ImagejRoi:
        """Return ImagejRoi instance from bytes.

        Parameters:
            data: Bytes in ImageJ ROI format.
            min_int_coord: Minimum integer coordinate for unwrapping.

        Returns:
            ImagejRoi instance decoded from bytes.

        """
        if len(data) < 64:
            msg = f'ImageJ ROI data too short: {len(data)} < 64 bytes'
            raise ValueError(msg)
        if data[:4] != b'Iout':
            msg = f'not an ImageJ ROI {data[:4]!r}'
            raise ValueError(msg)

        self = cls()

        (
            self.version,
            roitype,
            self.top,
            self.left,
            self.bottom,
            self.right,
            self.n_coordinates,
            # skip 16 bytes: x1,y1,x2,y2 or x,y,width,height or size
            self.stroke_width,
            self.shape_roi_size,
            self.stroke_color,
            self.fill_color,
            subtype,
            options,
            self.arrow_style_or_aspect_ratio,
            self.arrow_head_size,
            self.rounded_rect_arc_size,
            self.position,
            header2_offset,
        ) = struct.unpack(
            self.byteorder + 'hBxhhhhH16xhi4s4shhBBhii', data[4:64]
        )

        min_int_coord = ImagejRoi.min_int_coord(min_int_coord)

        # unwrap bounding box coordinates that are clearly wrapped
        if self.top < min_int_coord:
            self.top += 65536
        if self.bottom < min_int_coord:
            self.bottom += 65536
        if self.left < min_int_coord:
            self.left += 65536
        if self.right < min_int_coord:
            self.right += 65536

        # Handle wrap-around in the ambiguous range [min_int_coord, 0]
        # Values in this range could be wrapped or genuine negatives
        # Unwrap if it would create a valid bounding box
        # (bottom > top, right > left)
        if min_int_coord <= self.bottom <= 0 and self.bottom <= self.top:
            self.bottom += 65536
        if min_int_coord <= self.right <= 0 and self.right <= self.left:
            self.right += 65536

        self.roitype = ROI_TYPE(roitype)
        self.subtype = ROI_SUBTYPE(subtype)
        self.options = ROI_OPTIONS(options)

        if self.subpixelrect:
            self.xd, self.yd, self.widthd, self.heightd = struct.unpack(
                self.byteorder + 'ffff', data[18:34]
            )
        elif self.roitype == ROI_TYPE.LINE or (
            self.roitype == ROI_TYPE.FREEHAND
            and self.subtype in {ROI_SUBTYPE.ELLIPSE, ROI_SUBTYPE.ROTATED_RECT}
        ):
            self.x1, self.y1, self.x2, self.y2 = struct.unpack(
                self.byteorder + 'ffff', data[18:34]
            )
        elif self.n_coordinates == 0:
            self.n_coordinates = struct.unpack(
                self.byteorder + 'i', data[18:22]
            )[0]

        if 0 < header2_offset < len(data) - 52:
            (
                self.c_position,
                self.z_position,
                self.t_position,
                name_offset,
                name_length,
                self.overlay_label_color,
                self.overlay_font_size,
                self.group,
                self.image_opacity,
                self.image_size,
                self.float_stroke_width,
                roi_props_offset,
                roi_props_length,
                counters_offset,
            ) = struct.unpack(
                self.byteorder + '4xiiiii4shBBifiii',
                data[header2_offset : header2_offset + 52],
            )

            # handle extended group for version >= 229 (groups > 255)
            if self.version >= 229 and self.group == 0:
                group_offset = header2_offset + 52
                if group_offset + 2 <= len(data):
                    self.group = struct.unpack(
                        self.byteorder + 'H',
                        data[group_offset : group_offset + 2],
                    )[0]

            if name_offset > 0 and name_length > 0:
                name_end = name_offset + name_length * 2
                if name_end <= len(data):
                    name = data[name_offset:name_end]
                    self.name = name.decode(self.utf16)
                else:
                    logger().warning(
                        f'ImagejRoi name exceeds data size: '
                        f'{name_end} > {len(data)}'
                    )

            if roi_props_offset > 0 and roi_props_length > 0:
                props_end = roi_props_offset + roi_props_length * 2
                if props_end <= len(data):
                    props = data[roi_props_offset:props_end]
                    self.props = props.decode(self.utf16)
                else:
                    logger().warning(
                        f'ImagejRoi props exceeds data size: '
                        f'{props_end} > {len(data)}'
                    )

            if counters_offset > 0:
                counters: NDArray[numpy.uint32] = numpy.ndarray(
                    shape=self.n_coordinates,
                    dtype=self.byteorder + 'u4',
                    buffer=data,
                    offset=counters_offset,
                )
                self.counters = (counters & 0xFF).astype(numpy.uint8)
                self.counter_positions = (counters >> 8).astype(
                    numpy.uint32, copy=False
                )

        if self.version >= 218 and self.subtype == ROI_SUBTYPE.TEXT:
            (
                self.text_size,
                style_and_justification,
                name_length,
                text_length,
            ) = struct.unpack(self.byteorder + 'iiii', data[64:80])
            self.text_style = style_and_justification & 255
            self.text_justification = (style_and_justification >> 8) & 3
            off = 80
            self.text_name = data[off : off + name_length * 2].decode(
                self.utf16
            )
            off += name_length * 2
            self.text = data[off : off + text_length * 2].decode(self.utf16)
            if self.version >= 225:
                off += text_length * 2
                self.text_angle = struct.unpack(
                    self.byteorder + 'f', data[off : off + 4]
                )[0]

        elif self.version >= 221 and self.subtype == ROI_SUBTYPE.IMAGE:
            if 0 < self.image_size <= len(data) - 64:
                self.image_data = data[64 : 64 + self.image_size]
            else:
                logger().warning(
                    'ImagejRoi image data invalid: '
                    f'size={self.image_size}, data length={len(data)}'
                )

        elif self.roitype in (
            ROI_TYPE.POLYGON,
            ROI_TYPE.FREEHAND,
            ROI_TYPE.TRACED,
            ROI_TYPE.POLYLINE,
            ROI_TYPE.FREELINE,
            ROI_TYPE.ANGLE,
            ROI_TYPE.POINT,
        ):
            self.integer_coordinates = numpy.ndarray(
                shape=(self.n_coordinates, 2),
                dtype=self.byteorder + 'i2',
                buffer=data,
                offset=64,
                order='F',
            ).astype(numpy.int32)

            # unwrap negative integer_coordinates (wrapped uint16 values)
            select = self.integer_coordinates < 0
            self.integer_coordinates[select] += 65536

            if self.subpixelresolution:
                self.subpixel_coordinates = numpy.ndarray(
                    shape=(self.n_coordinates, 2),
                    dtype=self.byteorder + 'f4',
                    buffer=data,
                    offset=64 + self.n_coordinates * 4,
                    order='F',
                ).copy()

        elif self.composite and self.roitype == ROI_TYPE.RECT:
            self.multi_coordinates = numpy.ndarray(
                shape=self.shape_roi_size,
                dtype=self.byteorder + 'f4',
                buffer=data,
                offset=64,
            ).copy()

        elif self.roitype not in (ROI_TYPE.RECT, ROI_TYPE.LINE, ROI_TYPE.OVAL):
            logger().warning(f'cannot handle ImagejRoi type {self.roitype!r}')

        return self

    def tofile(
        self,
        filename: os.PathLike[Any] | str,
        /,
        *,
        name: str | None = None,
        mode: Literal['w', 'x', 'a'] | None = None,
    ) -> None:
        """Write ImagejRoi to ROI or ZIP file.

        Existing ZIP files are opened for append.

        Parameters:
            filename: Path to output .roi or .zip file.
            name: ROI name for ZIP file entry.
            mode: File mode {'w', 'x', or 'a'}.

        """
        filename = os.fspath(filename)
        if filename[-4:].lower() == '.zip':
            if name is None:
                name = self.name or self.autoname
            if name[-4:].lower() != '.roi':
                name += '.roi'
            if mode is None:
                mode = 'a' if os.path.exists(filename) else 'w'
            import zipfile

            assert mode is not None
            with (
                zipfile.ZipFile(filename, mode) as zf,
                zf.open(name, 'w') as fh,
            ):
                fh.write(self.tobytes())
        else:
            with open(filename, 'wb') as fh:
                fh.write(self.tobytes())

    def tobytes(self) -> bytes:
        """Return ImagejRoi as bytes.

        Returns:
            Bytes in ImageJ ROI format.

        """
        result = [b'Iout']

        result.append(
            struct.pack(
                self.byteorder + 'hBxhhhhH',
                self.version,
                self.roitype.value,
                numpy.array(self.top).astype(numpy.int16),
                numpy.array(self.left).astype(numpy.int16),
                numpy.array(self.bottom).astype(numpy.int16),
                numpy.array(self.right).astype(numpy.int16),
                self.n_coordinates if self.n_coordinates < 2**16 else 0,
            )
        )

        if self.subpixelrect:
            result.append(
                struct.pack(
                    self.byteorder + 'ffff',
                    self.xd,
                    self.yd,
                    self.widthd,
                    self.heightd,
                )
            )
        elif self.roitype == ROI_TYPE.LINE or (
            self.roitype == ROI_TYPE.FREEHAND
            and self.subtype in {ROI_SUBTYPE.ELLIPSE, ROI_SUBTYPE.ROTATED_RECT}
        ):
            result.append(
                struct.pack(
                    self.byteorder + 'ffff', self.x1, self.y1, self.x2, self.y2
                )
            )
        elif self.n_coordinates >= 2**16:
            result.append(
                struct.pack(self.byteorder + 'i12x', self.n_coordinates)
            )
        else:
            result.append(b'\x00' * 16)

        result.append(
            struct.pack(
                self.byteorder + 'hi4s4shhBBhi',
                self.stroke_width,
                self.shape_roi_size,
                self.stroke_color,
                self.fill_color,
                self.subtype.value,
                self.options.value,
                self.arrow_style_or_aspect_ratio,
                self.arrow_head_size,
                self.rounded_rect_arc_size,
                self.position,
            )
        )

        extradata = b''

        if self.version >= 218 and self.subtype == ROI_SUBTYPE.TEXT:
            style_and_justification = self.text_style
            style_and_justification |= self.text_justification << 8
            extradata = struct.pack(
                self.byteorder + 'iiii',
                self.text_size,
                style_and_justification,
                len(self.text_name),
                len(self.text),
            )
            extradata += self.text_name.encode(self.utf16)
            extradata += self.text.encode(self.utf16)
            extradata += struct.pack(self.byteorder + 'f', self.text_angle)

        elif self.version >= 221 and self.subtype == ROI_SUBTYPE.IMAGE:
            if self.image_data is not None:
                extradata = self.image_data
            else:
                logger().warning(
                    'ImagejRoi IMAGE subtype but image_data is None'
                )

        elif self.roitype in (
            ROI_TYPE.POLYGON,
            ROI_TYPE.FREEHAND,
            ROI_TYPE.TRACED,
            ROI_TYPE.POLYLINE,
            ROI_TYPE.FREELINE,
            ROI_TYPE.ANGLE,
            ROI_TYPE.POINT,
        ):
            if self.integer_coordinates is not None:
                if self.integer_coordinates.shape != (self.n_coordinates, 2):
                    msg = (
                        'integer_coordinates.shape '
                        f'{self.integer_coordinates.shape} '
                        f'!= ({self.n_coordinates}, 2)'
                    )
                    raise ValueError(msg)
                coord = self.integer_coordinates.astype(
                    self.byteorder + 'i2', copy=False
                )
                extradata = coord.tobytes(order='F')
            if self.subpixel_coordinates is not None:
                if self.subpixel_coordinates.shape != (self.n_coordinates, 2):
                    msg = (
                        'subpixel_coordinates.shape '
                        f'{self.subpixel_coordinates.shape} '
                        f'!= ({self.n_coordinates}, 2)'
                    )
                    raise ValueError(msg)
                coord = self.subpixel_coordinates.astype(
                    self.byteorder + 'f4', copy=False
                )
                extradata += coord.tobytes(order='F')

        elif self.composite and self.roitype == ROI_TYPE.RECT:
            assert self.multi_coordinates is not None
            coord = self.multi_coordinates.astype(
                self.byteorder + 'f4', copy=False
            )
            extradata += coord.tobytes()

        header2_offset = 64 + len(extradata)
        result.append(struct.pack(self.byteorder + 'i', header2_offset))
        result.append(extradata)

        offset = header2_offset + 64
        name_length = len(self.name)
        name_offset = offset if name_length > 0 else 0
        offset += name_length * 2
        roi_props_length = len(self.props)
        roi_props_offset = offset if roi_props_length > 0 else 0
        offset += roi_props_length * 2
        counters_offset = offset if self.counters is not None else 0

        # determine group byte value (use 0 if extended group will be written)
        group_byte = (
            0 if self.version >= 229 and self.group > 255 else self.group
        )

        result.append(
            struct.pack(
                self.byteorder + '4xiiiii4shBBifiii',
                self.c_position,
                self.z_position,
                self.t_position,
                name_offset,
                name_length,
                self.overlay_label_color,
                self.overlay_font_size,
                group_byte,
                self.image_opacity,
                self.image_size,
                self.float_stroke_width,
                roi_props_offset,
                roi_props_length,
                counters_offset,
            )
        )

        # write extended group for version >= 229 if group > 255
        if self.version >= 229 and self.group > 255:
            result.append(struct.pack(self.byteorder + 'H', self.group))
            result.append(b'\x00' * 10)  # 10 bytes padding
        else:
            result.append(b'\x00' * 12)  # 12 bytes padding

        if name_length > 0:
            result.append(self.name.encode(self.utf16))
        if roi_props_length > 0:
            result.append(self.props.encode(self.utf16))
        if self.counters is not None:
            counters = numpy.asarray(
                self.counters, dtype=self.byteorder + 'u4'
            )
            if self.counter_positions is not None:
                indices = numpy.asarray(
                    self.counter_positions, dtype=self.byteorder + 'u4'
                )
                counters = counters & 0xFF | indices << 8
                counters = counters.astype(self.byteorder + 'u4', copy=False)
            result.append(counters.tobytes())

        return b''.join(result)

    def plot(
        self,
        ax: Axes | None = None,
        *,
        rois: Iterable[ImagejRoi] | None = None,
        title: str | None = None,
        bounds: bool = False,
        invert_yaxis: bool | None = None,
        show: bool = True,
        **kwargs: Any,
    ) -> None:
        """Plot draft of coordinates using matplotlib.

        Parameters:
            ax: Matplotlib axes to plot on. Create new figure if None.
            rois: Multiple ROIs to plot together.
            title: Figure title.
            bounds: Show bounding rectangle.
            invert_yaxis: Invert Y axis. Auto-determined if None.
            show: Call pyplot.show().
            **kwargs: Additional arguments passed to matplotlib plot functions.

        """
        fig: Any
        roitype = self.roitype
        subtype = self.subtype

        if ax is None:
            from matplotlib import pyplot
            from matplotlib.patches import Rectangle

            fig, ax = pyplot.subplots()
            ax.set_aspect('equal')
            ax.set_facecolor((0.8, 0.8, 0.8))
            if title is None:
                fig.suptitle(f'{self.name!r}')
            else:
                fig.suptitle(title)
            if bounds and rois is None:
                ax.set_title(f'{roitype.name} {subtype.name}')
                ax.add_patch(
                    Rectangle(
                        (self.left, self.top),
                        self.right - self.left,
                        self.bottom - self.top,
                        linewidth=1,
                        edgecolor='0.9',
                        facecolor='none',
                    )
                )
            if invert_yaxis is None:
                invert_yaxis = True
        else:
            fig = None
            if invert_yaxis is None:
                invert_yaxis = False

        if rois is not None:
            for roi in rois:
                roi.plot(ax=ax, **kwargs)
            if invert_yaxis:
                ax.invert_yaxis()
            if show:
                pyplot.show()
            return

        if 'color' not in kwargs and 'c' not in kwargs:
            kwargs['color'] = self.hexcolor(self.stroke_color)
        if 'linewidth' not in kwargs and 'lw' not in kwargs:
            # TODO: use data units
            if self.float_stroke_width > 0.0:
                kwargs['linewidth'] = self.float_stroke_width
            elif self.stroke_width > 0:
                kwargs['linewidth'] = self.stroke_width
        if roitype == ROI_TYPE.POINT:
            if 'marker' not in kwargs:
                # map point type to matplotlib marker
                if self.version >= 226:
                    match self.point_type:
                        case ROI_POINT_TYPE.HYBRID:
                            kwargs['marker'] = '+'  # no exact hybrid marker
                        case ROI_POINT_TYPE.DOT:
                            kwargs['marker'] = 'o'
                        case ROI_POINT_TYPE.CIRCLE:
                            kwargs['marker'] = 'o'
                            kwargs['markerfacecolor'] = 'none'
                        case ROI_POINT_TYPE.CROSS:
                            kwargs['marker'] = '+'
                        case _:
                            kwargs['marker'] = 'x'
                else:
                    kwargs['marker'] = 'x'
            if 'linestyle' not in kwargs:
                kwargs['linestyle'] = ''
            # use point_size if available (version 226+)
            if (
                'markersize' not in kwargs
                and 'ms' not in kwargs
                and self.version >= 226
                and self.point_size > 0
            ):
                kwargs['markersize'] = self.point_size * 2

        match (roitype, subtype):
            case (ROI_TYPE.LINE, ROI_SUBTYPE.ARROW):
                line = self.coordinates()
                x, y = line[0]
                dx, dy = line[1] - line[0]
                if 'head_width' not in kwargs and self.arrow_head_size > 0:
                    kwargs['head_width'] = self.arrow_head_size
                kwargs['length_includes_head'] = True
                ax.arrow(x, y, dx, dy, **kwargs)
                if self.options & ROI_OPTIONS.DOUBLE_HEADED:
                    x, y = line[1]
                    ax.arrow(x, y, -dx, -dy, **kwargs)
            case (ROI_TYPE.RECT, ROI_SUBTYPE.TEXT):
                coordslist = self.coordinates(multi=True)
                if coordslist and len(coordslist[0]) >= 3:
                    coords = coordslist[0]
                    if 'fontsize' not in kwargs and self.text_size > 0:
                        kwargs['fontsize'] = self.text_size
                    text = ax.text(
                        coords[1][0],
                        coords[1][1],
                        self.text,
                        va='center_baseline',
                        rotation=self.text_angle,
                        rotation_mode='anchor',
                        **kwargs,
                    )
                    scale_text(text, width=abs(coords[2, 0] - coords[0, 0]))
                # ax.plot(
                #     coords[:, 0],
                #     coords[:, 1],
                #     linewidth=1,
                #     color=kwargs.get('color', 0.9),
                #     ls=':',
                # )
            case (ROI_TYPE.RECT, ROI_SUBTYPE.IMAGE):
                if self.image is not None:
                    alpha = self.image_opacity / 255.0
                    ax.imshow(
                        self.image,
                        extent=(self.left, self.right, self.bottom, self.top),
                        alpha=alpha if alpha > 0 else 1.0,
                        origin='upper',
                        interpolation='nearest',
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k in {'cmap', 'vmin', 'vmax'}
                        },
                    )
            case _:
                for coords in self.coordinates(multi=True):
                    ax.plot(coords[:, 0], coords[:, 1], **kwargs)

        # integer limits might be bogus
        if (
            self.left < self.right
            and self.top < self.bottom
            and self.left > -256
            and self.left < 24576
            and self.bottom > -256
            and self.bottom < 24576
            and self.right > -256
            and self.right < 24576
            and self.top > -256
            and self.top < 24576
        ):
            ax.plot(self.left, self.bottom, '')
            ax.plot(self.right, self.top, '')

        if invert_yaxis:
            ax.invert_yaxis()

        if show and fig is not None:
            pyplot.show()

    def coordinates(
        self,
        *,
        multi: bool = False,
    ) -> NDArray[Any] | list[NDArray[Any]]:
        """Return x, y coordinates as numpy array for display.

        Parameters:
            multi: Return list of coordinate arrays for composite shapes.

        Returns:
            Array of shape (n, 2) with x, y coordinates,
            or list of such arrays if multi=True.

        """
        coords: Any
        if self.subpixel_coordinates is not None:
            coords = self.subpixel_coordinates.copy()
        elif self.integer_coordinates is not None:
            coords = self.integer_coordinates + [  # noqa: RUF005
                self.left,
                self.top,
            ]
        elif self.multi_coordinates is not None:
            coordslist = self.path2coords(self.multi_coordinates)
            if not multi:
                return sorted(coordslist, key=lambda x: x.size)[-1]
            return coordslist
        elif self.roitype == ROI_TYPE.LINE:
            coords = numpy.array(
                [[self.x1, self.y1], [self.x2, self.y2]], numpy.float32
            )
        elif self.roitype == ROI_TYPE.OVAL:
            coords = oval([[self.left, self.top], [self.right, self.bottom]])
        elif self.roitype == ROI_TYPE.RECT:
            coords = numpy.array(
                [
                    [self.left, self.top],
                    [self.left, self.bottom],
                    [self.right, self.bottom],
                    [self.right, self.top],
                    [self.left, self.top],
                ],
                numpy.float32,
            )
        else:
            coords = numpy.empty((0, 2), dtype=self.byteorder + 'i4')
        return [coords] if multi else coords

    def hexcolor(self, b: bytes, /, default: str | None = None) -> str | None:
        """Return color (bytes) as hex triplet or default if black.

        Parameters:
            b: Color as 4 ARGB bytes.
            default: Value to return if color is black/none.

        Returns:
            Hex color string like '#rrggbb', or default if black.

        """
        if b == ROI_COLOR_NONE:
            return default
        if len(b) != 4:
            msg = f'color bytes must be length 4, got {len(b)}'
            raise ValueError(msg)
        if self.byteorder == '>':
            return f'#{b[1]:02x}{b[2]:02x}{b[3]:02x}'
        return f'#{b[3]:02x}{b[2]:02x}{b[1]:02x}'

    @staticmethod
    def path2coords(
        multi_coordinates: NDArray[numpy.float32], /
    ) -> list[NDArray[numpy.float32]]:
        """Return list of coordinate arrays from 2D geometric path.

        Parameters:
            multi_coordinates: Path data with MOVETO, LINETO, CLOSE operations.

        Returns:
            List of coordinate arrays, one per path segment.

        """
        coordinates: list[NDArray[numpy.float32]] = []
        points: list[tuple[float, float]] = []
        path: list[float] = multi_coordinates.tolist()
        n = 0
        m = 0

        if not path:
            return coordinates

        while n < len(path):
            op = int(path[n])
            if op == 0:
                # MOVETO
                if n > 0:
                    coordinates.append(
                        numpy.array(points, dtype=numpy.float32)
                    )
                    points = []
                points.append((path[n + 1], path[n + 2]))
                m = len(points) - 1
                n += 3
            elif op == 1:
                # LINETO
                points.append((path[n + 1], path[n + 2]))
                n += 3
            elif op == 4:
                # CLOSE
                if not points:
                    msg = 'CLOSE operation without any points'
                    raise RuntimeError(msg)
                points.append(points[m])
                n += 1
            elif op == 2 or op == 3:  # noqa: PLR1714
                # QUADTO or CUBICTO
                msg = f'PathIterator command {op!r} not supported'
                raise NotImplementedError(msg)
            else:
                msg = f'invalid PathIterator command {op!r}'
                raise RuntimeError(msg)

        if points:
            coordinates.append(numpy.array(points, dtype=numpy.float32))
        return coordinates

    @staticmethod
    def min_int_coord(value: int | None = None, /) -> int:
        """Return minimum integer coordinate value.

        The default, -5000, is used by ImageJ.
        A value of -32768 means to use int16 range, 0 means uint16 range.

        Parameters:
            value: Minimum coordinate value (-32768 to 0), or None for default.

        Returns:
            Minimum integer coordinate value (-5000 default).

        """
        if value is None:
            return -5000
        if -32768 <= value <= 0:
            return int(value)
        msg = f'{value=} out of range'
        raise ValueError(msg)

    @property
    def composite(self) -> bool:
        """ROI is composite shape."""
        return self.shape_roi_size > 0

    @property
    def subpixelresolution(self) -> bool:
        """ROI has subpixel resolution."""
        return self.version >= 222 and bool(
            self.options & ROI_OPTIONS.SUB_PIXEL_RESOLUTION
        )

    @property
    def drawoffset(self) -> bool:
        """ROI has draw offset."""
        return self.subpixelresolution and bool(
            self.options & ROI_OPTIONS.DRAW_OFFSET
        )

    @property
    def subpixelrect(self) -> bool:
        """ROI has subpixel rectangle."""
        return (
            self.version >= 223
            and self.subpixelresolution
            and self.roitype in (ROI_TYPE.RECT, ROI_TYPE.OVAL)
        )

    @property
    def autoname(self) -> str:
        """Name generated from positions."""
        y = (self.bottom - self.top) // 2
        x = (self.right - self.left) // 2
        name = f'{y:05}-{x:05}'
        if self.counter_positions is not None:
            tzc = int(self.counter_positions.max())
            name = f'{tzc:05}-' + name
        return name

    @property
    def point_type(self) -> ROI_POINT_TYPE:
        """Point type for POINT ROIs (version 226+).

        Maps to arrow_style_or_aspect_ratio field.
        Only meaningful for ROI_TYPE.POINT.

        """
        return ROI_POINT_TYPE(self.arrow_style_or_aspect_ratio)

    @point_type.setter
    def point_type(self, value: int | ROI_POINT_TYPE, /) -> None:
        if not isinstance(value, ROI_POINT_TYPE):
            value = ROI_POINT_TYPE(value)
        self.arrow_style_or_aspect_ratio = value.value

    @property
    def point_size(self) -> ROI_POINT_SIZE:
        """Point size for POINT ROIs (version 226+).

        Maps to stroke_width field. Only meaningful for ROI_TYPE.POINT.

        """
        return ROI_POINT_SIZE(self.stroke_width)

    @point_size.setter
    def point_size(self, value: int | ROI_POINT_SIZE, /) -> None:
        if not isinstance(value, ROI_POINT_SIZE):
            value = ROI_POINT_SIZE(value)
        self.stroke_width = value.value

    @property
    def image(self) -> NDArray[Any] | None:
        """Decoded image as numpy array for IMAGE subtype ROIs.

        Image is None if no image data is stored in file or decoding failed.

        """
        if self.image_data is None:
            return None
        try:
            # TODO: is image data always TIFF?
            from imagecodecs import tiff_decode

            return tiff_decode(self.image_data)
        except Exception as exc:
            logger().warning(f'ImagejRoi failed to decode image data: {exc}')
        return None

    @image.setter
    def image(self, value: NDArray[Any] | None, /) -> None:
        if value is None:
            self.image_data = None
            self.image_size = 0
            return

        if value.ndim == 3:
            if value.shape[2] not in {3, 4}:
                msg = 'RGB image must have 3 or 4 channels'
                raise ValueError(msg)
            if value.dtype.char != 'B':
                msg = f'invalid RGB dtype={value.dtype} != uint8'
                raise ValueError(msg)
        elif value.ndim == 2:
            if value.dtype.char not in {'B', 'H', 'f'}:
                msg = f'invalid grayscale dtype={value.dtype}'
                raise ValueError(msg)
        else:
            msg = 'image array must be 2D (grayscale) or 3D (RGB/RGBA)'
            raise ValueError(msg)

        self.right = self.left + value.shape[1]
        self.bottom = self.top + value.shape[0]

        from imagecodecs import tiff_encode

        encoded = tiff_encode(value, description='ImageJ=1.11a name=')
        assert isinstance(encoded, bytes)
        self.image_data = encoded
        self.image_size = len(self.image_data)

    @property
    def properties(self) -> dict[str, Any]:
        """Decoded props field as dictionary."""
        val: Any
        props = {}
        for line in self.props.splitlines():
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                if val == 'true':
                    val = True
                elif val == 'false':
                    val = False
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        with contextlib.suppress(ValueError):
                            val = float(val)
                props[key] = val
        return props

    @properties.setter
    def properties(self, value: dict[str, Any], /) -> None:
        lines = []
        for item in sorted(value.items()):
            key, val = item
            if isinstance(val, bool):
                val = 'true' if val else 'false'
            # TODO: does float need specific format?
            lines.append(f'{key}: {val}\n')
        self.props = ''.join(lines)

    @property
    def utf16(self) -> str:
        """UTF-16 codec depending on byteorder."""
        return 'utf-16' + ('be' if self.byteorder == '>' else 'le')

    def __hash__(self) -> int:
        """Return hash of ImagejRoi."""
        return hash(
            (
                self.tobytes(),
                self.left,
                self.top,
                self.right,
                self.bottom,
            )
        )

    def __eq__(self, other: object) -> bool:
        """Return True if two ImagejRoi are the same."""
        return (
            isinstance(other, ImagejRoi)
            and self.tobytes() == other.tobytes()
            and self.left == other.left
            and self.top == other.top
            and self.right == other.right
            and self.bottom == other.bottom
        )

    def __repr__(self) -> str:
        info = [f'{self.__class__.__name__}(']
        for name, value in self.__dict__.items():
            if isinstance(value, numpy.ndarray):
                v = repr(value).replace('    ', ' ')
                v = v.replace('([[', '([\n    [')
                info.append(f'{name}=numpy.{v},')
            elif value == getattr(ImagejRoi, name):
                pass
            elif isinstance(value, enum.Enum):
                info.append(f'{name}={enumstr(value)},')
            else:
                info.append(f'{name}={value!r},')
        return indent(*info, end='\n)')


def scale_text(
    text: Any,
    width: float,
    *,
    offset: tuple[float, float] | None = None,
) -> None:
    """Scale matplotlib text to width in data coordinates.

    Parameters:
        text: Matplotlib text object to scale.
        width: Target width in data coordinates.
        offset: Optional offset tuple (x, y).

    """
    from matplotlib.patheffects import AbstractPathEffect
    from matplotlib.transforms import Bbox

    class TextScaler(AbstractPathEffect):
        def __init__(
            self,
            text: Any,
            width: float,
            offset: tuple[float, float] | None = None,
        ) -> None:
            if offset is None:
                offset = (0.0, 0.0)
            super().__init__(offset)
            self._text = text
            self._width = width

        def draw_path(
            self,
            renderer: Any,
            gc: Any,
            tpath: Any,
            affine: Any,
            rgbFace: Any = None,  # noqa: N803
        ) -> None:
            ax = self._text.axes
            renderer = ax.get_figure().canvas.get_renderer()
            bbox = text.get_window_extent(renderer=renderer)
            bbox = Bbox(ax.transData.inverted().transform(bbox))
            if self._width > 0 and bbox.width > 0:
                scale = self._width / bbox.width
                affine = affine.from_values(scale, 0, 0, scale, 0, 0) + affine
            renderer.draw_path(gc, tpath, affine, rgbFace)

    text.set_path_effects([TextScaler(text, width, offset)])


def oval(rect: ArrayLike, /, points: int = 33) -> NDArray[numpy.float32]:
    """Return coordinates of oval inscribed in bounding rectangle.

    Parameters:
        rect: Bounding rectangle as [[left, top], [right, bottom]].
        points: Number of points to generate around oval.

    Returns:
        Array of shape (points, 2) with x, y coordinates.

    """
    arr = numpy.asarray(rect, dtype=numpy.float32)
    c = numpy.linspace(0.0, 2.0 * numpy.pi, num=points, dtype=numpy.float32)
    c = numpy.array([numpy.cos(c), numpy.sin(c)]).T
    r = arr[1] - arr[0]
    r /= 2.0
    c *= r
    c += arr[0] + r
    return c


def indent(*args: Any, sep: str = '', end: str = '') -> str:
    """Return joined string representations of objects with indented lines.

    Parameters:
        *args: Objects to join and indent.
        sep: Separator between objects.
        end: String appended at end.

    Returns:
        Indented string representation.

    """
    text: str = (sep + '\n').join(
        arg if isinstance(arg, str) else repr(arg) for arg in args
    )
    return (
        '\n'.join(
            ('    ' + line if line else line)
            for line in text.splitlines()
            if line
        )[4:]
        + end
    )


def enumstr(v: enum.Enum | None, /) -> str:
    """Return IntEnum or IntFlag as str.

    Parameters:
        v: Enum value to convert to string.

    Returns:
        String representation of enum value.

    """
    # repr() and str() of enums are type, value, and version dependent
    if v is None:
        return 'None'
    # return f'{v.__class__.__name__}({v.value})'
    s = repr(v)
    s = s[1:].split(':', 1)[0]
    if 'UNKNOWN' in s:
        s = f'{v.__class__.__name__}({v.value})'
    elif '|' in s:
        # IntFlag combination
        s = s.replace('|', ' | ' + v.__class__.__name__ + '.')
    elif not hasattr(v, 'name') or v.name is None:
        s = f'{v.__class__.__name__}({v.value})'
    return s


def logger() -> logging.Logger:
    """Return logger for roifile module.

    Returns:
        Logger instance for 'roifile' module.

    """
    return logging.getLogger('roifile')


def main(argv: list[str] | None = None) -> int:
    """Roifile command line usage main function.

    Show all ImageJ ROIs in file or all files in directory::

        python -m roifile file_or_directory

    Parameters:
        argv: Command line arguments. Uses sys.argv if None.

    Returns:
        Exit code (0 for success).

    """
    from glob import glob

    if argv is None:
        argv = sys.argv

    if len(argv) == 1:
        files = glob('*.roi')
        files += glob('*.zip')
        files += glob('*.tif')
    elif '*' in argv[1]:
        files = glob(argv[1])
    elif os.path.isdir(argv[1]):
        files = []
        for ext in ('roi', 'zip', 'tif'):
            files += glob(f'{argv[1]}/*.{ext}')
    else:
        files = argv[1:]

    for filename in files:
        print(filename)  # noqa: T201
        try:
            rois = ImagejRoi.fromfile(filename)
            title = os.path.split(filename)[-1]
            if isinstance(rois, list):
                for roi in rois:
                    print(roi, '\n')  # noqa: T201
                    if sys.flags.dev_mode:
                        assert roi == ImagejRoi.frombytes(roi.tobytes())
                if rois:
                    rois[0].plot(rois=rois, title=title)
            else:
                print(rois, '\n')  # noqa: T201
                if sys.flags.dev_mode:
                    assert rois == ImagejRoi.frombytes(rois.tobytes())
                rois.plot(title=title)
        except ValueError as exc:
            if sys.flags.dev_mode:
                raise
            print(filename, exc)  # noqa: T201
            continue
    return 0


if __name__ == '__main__':
    sys.exit(main())
