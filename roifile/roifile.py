# roifile.py

# Copyright (c) 2020-2024, Christoph Gohlke
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
:License: BSD 3-Clause
:Version: 2024.3.20
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

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.8, 3.12.2
- `Numpy <https://pypi.org/project/numpy/>`_ 1.26.4
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2024.2.12 (optional)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.8.3 (optional)

Revisions
---------

2024.3.20

- Fix writing generator of ROIs (#9).

2024.1.10

- Support text rotation.
- Improve text rendering.
- Avoid array copies.
- Limit size read from files.

2023.8.30

- Fix linting issues.
- Add py.typed marker.

2023.5.12

- Improve object repr and type hints.
- Drop support for Python 3.8 and numpy < 1.21 (NEP29).

2023.2.12

- Delay import of zipfile.
- Verify shape of coordinates on write.

2022.9.19

- Fix integer coordinates to -5000..60536 conforming with ImageJ (breaking).
- Add subpixel_coordinates in frompoints for out-of-range integer coordinates.

2022.7.29

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
- `napari_jroitools <https://github.com/jayunruh/napari_jroitools>`_

Examples
--------

Create a new ImagejRoi instance from an array of x, y coordinates:

>>> roi = ImagejRoi.frompoints([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
>>> roi.roitype = ROI_TYPE.POINT
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

Plot the ROI using matplotlib:

>>> roi.plot()

View the overlays stored in a ROI, ZIP, or TIFF file from a command line::

    python -m roifile _test.roi

"""

from __future__ import annotations

__version__ = '2024.3.20'

__all__ = [
    'roiread',
    'roiwrite',
    'ImagejRoi',
    'ROI_TYPE',
    'ROI_SUBTYPE',
    'ROI_OPTIONS',
    'ROI_POINT_TYPE',
    'ROI_POINT_SIZE',
    'ROI_COLOR_NONE',
]

import dataclasses
import enum
import logging
import os
import struct
import sys
import uuid
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Literal

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
    mode: Literal['r', 'w', 'x', 'a'] | None = None,
) -> None:
    """Write ImagejRoi instance(s) to ROI or ZIP file.

    Write an ImagejRoi instance to a ROI file or write a sequence of ImagejRoi
    instances to a ZIP file. Existing ZIP files are opened for append.

    """
    filename = os.fspath(filename)

    if isinstance(roi, ImagejRoi):
        assert name is None or isinstance(name, str)
        return roi.tofile(filename, name=name, mode=mode)

    if mode is None:
        mode = 'a' if os.path.exists(filename) else 'w'

    if name is not None:
        if isinstance(name, str):
            raise ValueError("'name' is not an iterable of str")
        name = iter(name)

    import zipfile

    with zipfile.ZipFile(filename, mode) as zf:
        for r in roi:
            if name is None:
                n = r.name if r.name else r.autoname
            else:
                n = next(name)
            n = n if n[-4:].lower() == '.roi' else n + '.roi'
            with zf.open(n, 'w') as fh:
                fh.write(r.tobytes())
    return None


class ROI_TYPE(enum.IntEnum):
    """ImageJ ROI types."""

    POLYGON = 0
    RECT = 1
    OVAL = 2
    LINE = 3
    FREELINE = 4
    POLYLINE = 5
    NOROI = 6
    FREEHAND = 7
    TRACED = 8
    ANGLE = 9
    POINT = 10


class ROI_SUBTYPE(enum.IntEnum):
    """ImageJ ROI subtypes."""

    UNDEFINED = 0
    TEXT = 1
    ARROW = 2
    ELLIPSE = 3
    IMAGE = 4
    ROTATED_RECT = 5


class ROI_OPTIONS(enum.IntFlag):
    """ImageJ ROI options."""

    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256
    ZERO_TRANSPARENT = 512
    SHOW_LABELS = 1024
    SCALE_LABELS = 2048
    PROMPT_BEFORE_DELETING = 4096


class ROI_POINT_TYPE(enum.IntEnum):
    """ImageJ ROI point types."""

    HYBRID = 0
    CROSS = 1
    # CROSSHAIR = 1
    DOT = 2
    CIRCLE = 3


class ROI_POINT_SIZE(enum.IntEnum):
    """ImageJ ROI point sizes."""

    TINY = 1
    SMALL = 3
    MEDIUM = 5
    LARGE = 7
    EXTRA_LARGE = 11
    XXL = 17
    XXXL = 25


ROI_COLOR_NONE = b'\x00\x00\x00\x00'
"""No color or Black."""


@dataclasses.dataclass
class ImagejRoi:
    """Read and write ImageJ ROI format."""

    byteorder: Literal['>', '<'] = '>'
    roitype: ROI_TYPE = ROI_TYPE.POLYGON
    subtype: ROI_SUBTYPE = ROI_SUBTYPE.UNDEFINED
    options: ROI_OPTIONS = ROI_OPTIONS(0)
    name: str = ''
    props: str = ''
    version: int = 217
    top: int = 0
    left: int = 0
    bottom: int = 0
    right: int = 0
    n_coordinates: int = 0
    stroke_width: int = 0
    shape_roi_size: int = 0
    stroke_color: bytes = ROI_COLOR_NONE
    fill_color: bytes = ROI_COLOR_NONE
    arrow_style_or_aspect_ratio: int = 0
    arrow_head_size: int = 0
    rounded_rect_arc_size: int = 0
    position: int = 0
    c_position: int = 0
    z_position: int = 0
    t_position: int = 0
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    xd: float = 0.0
    yd: float = 0.0
    widthd: float = 0.0
    heightd: float = 0.0
    overlay_label_color: bytes = ROI_COLOR_NONE
    overlay_font_size: int = 0
    group: int = 0
    image_opacity: int = 0
    image_size: int = 0
    float_stroke_width: float = 0.0
    text_size: int = 0
    text_style: int = 0
    text_justification: int = 0
    text_angle: float = 0.0
    text_name: str = ''
    text: str = ''
    counters: NDArray[numpy.uint8] | None = None
    counter_positions: NDArray[numpy.uint32] | None = None
    integer_coordinates: NDArray[numpy.int32] | None = None
    subpixel_coordinates: NDArray[numpy.float32] | None = None
    multi_coordinates: NDArray[numpy.float32] | None = None

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
    ) -> ImagejRoi:
        """Return ImagejRoi instance from sequence of Point coordinates.

        Use floating point coordinates for subpixel precision or values outside
        the range -5000..60536.

        A FREEHAND ROI with options OVERLAY_BACKGROUNDS and OVERLAY_LABELS is
        returned.

        """
        if points is None:
            return cls()

        self = cls()
        self.version = 226
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
        if coords.dtype.kind == 'f' or (
            numpy.any(coords > 60000) or numpy.any(coords < -5000)
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

        """
        filename = os.fspath(filename)
        if filename[-4:].lower() == '.tif':
            import tifffile

            with tifffile.TiffFile(filename) as tif:
                if tif.imagej_metadata is None:
                    raise ValueError('file does not contain ImagejRoi')
                rois = []
                if 'Overlays' in tif.imagej_metadata:
                    overlays = tif.imagej_metadata['Overlays']
                    if isinstance(overlays, list):
                        rois.extend(overlays)
                    else:
                        rois.append(overlays)
                if 'ROI' in tif.imagej_metadata:
                    roi = tif.imagej_metadata['ROI']
                    if isinstance(roi, list):
                        overlays.extend(roi)
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
        """Return ImagejRoi instance from bytes."""
        if data[:4] != b'Iout':
            raise ValueError(f'not an ImageJ ROI {data[:4]!r}')

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

        if self.top < min_int_coord:
            self.top += 65536
        if self.bottom < min_int_coord:
            self.bottom += 65536
        if self.bottom < 0 and self.bottom < self.top:
            self.bottom += 65536

        if self.left < min_int_coord:
            self.left += 65536
        if self.right < min_int_coord:
            self.right += 65536
        if self.right < 0 and self.right < self.left:
            self.right += 65536

        self.roitype = ROI_TYPE(roitype)
        self.subtype = ROI_SUBTYPE(subtype)
        self.options = ROI_OPTIONS(options)

        if self.subpixelrect:
            (self.xd, self.yd, self.widthd, self.heightd) = struct.unpack(
                self.byteorder + 'ffff', data[18:34]
            )
        elif (
            self.roitype == ROI_TYPE.LINE
            or self.roitype == ROI_TYPE.FREEHAND
            and self.subtype in {ROI_SUBTYPE.ELLIPSE, ROI_SUBTYPE.ROTATED_RECT}
        ):
            (self.x1, self.y1, self.x2, self.y2) = struct.unpack(
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

            if name_offset > 0 and name_length > 0:
                name = data[name_offset : name_offset + name_length * 2]
                self.name = name.decode(self.utf16)

            if roi_props_offset > 0 and roi_props_length > 0:
                props = data[
                    roi_props_offset : name_offset + roi_props_length * 2
                ]
                self.props = props.decode(self.utf16)

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

            select = self.integer_coordinates < min_int_coord
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
        name: str | None = None,
        mode: Literal['r', 'w', 'x', 'a'] | None = None,
    ) -> None:
        """Write ImagejRoi to ROI or ZIP file.

        Existing ZIP files are opened for append.

        """
        filename = os.fspath(filename)
        if filename[-4:].lower() == '.zip':
            if name is None:
                name = self.name if self.name else self.autoname
            if name[-4:].lower() != '.roi':
                name += '.roi'
            if mode is None:
                mode = 'a' if os.path.exists(filename) else 'w'
            import zipfile

            with zipfile.ZipFile(filename, mode) as zf:
                with zf.open(name, 'w') as fh:
                    fh.write(self.tobytes())
        else:
            with open(filename, 'wb') as fh:
                fh.write(self.tobytes())

    def tobytes(self) -> bytes:
        """Return ImagejRoi as bytes."""
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
        elif (
            self.roitype == ROI_TYPE.LINE
            or self.roitype == ROI_TYPE.FREEHAND
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
                    raise ValueError(
                        'integer_coordinates.shape '
                        f'{self.integer_coordinates.shape} '
                        f'!= ({self.n_coordinates}, 2)'
                    )
                coord = self.integer_coordinates.astype(
                    self.byteorder + 'i2', copy=False
                )
                extradata = coord.tobytes(order='F')
            if self.subpixel_coordinates is not None:
                if self.subpixel_coordinates.shape != (self.n_coordinates, 2):
                    raise ValueError(
                        'subpixel_coordinates.shape '
                        f'{self.subpixel_coordinates.shape} '
                        f'!= ({self.n_coordinates}, 2)'
                    )
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

        result.append(
            struct.pack(
                self.byteorder + '4xiiiii4shBBifiii12x',
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
            )
        )

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
        ax: Any | None = None,
        *,
        rois: Iterable[ImagejRoi] | None = None,
        title: str | None = None,
        bounds: bool = False,
        invert_yaxis: bool | None = None,
        **kwargs,
    ) -> None:
        """Plot a draft of coordinates using matplotlib."""
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
            pyplot.show()
            return

        if kwargs is None:
            kwargs = {}
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
                kwargs['marker'] = 'x'
            if 'linestyle' not in kwargs:
                kwargs['linestyle'] = ''

        if roitype == ROI_TYPE.LINE and subtype == ROI_SUBTYPE.ARROW:
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
        elif roitype == ROI_TYPE.RECT and subtype == ROI_SUBTYPE.TEXT:
            coords = self.coordinates(True)[0]
            if 'fontsize' not in kwargs and self.text_size > 0:
                kwargs['fontsize'] = self.text_size
            text = ax.text(
                *coords[1],
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
        else:
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

        if fig is not None:
            pyplot.show()

    def coordinates(
        self, multi: bool = False
    ) -> NDArray[Any] | list[NDArray[Any]]:
        """Return x, y coordinates as numpy array for display."""
        coords: Any
        if self.subpixel_coordinates is not None:
            coords = self.subpixel_coordinates.copy()
        elif self.integer_coordinates is not None:
            coords = self.integer_coordinates + [self.left, self.top]
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
        """Return color (bytes) as hex triplet or None if black."""
        if b == ROI_COLOR_NONE:
            return default
        if self.byteorder == '>':
            return f'#{b[1]:02x}{b[2]:02x}{b[3]:02x}'
        return f'#{b[3]:02x}{b[2]:02x}{b[1]:02x}'

    @staticmethod
    def path2coords(
        multi_coordinates: NDArray[numpy.float32], /
    ) -> list[NDArray[numpy.float32]]:
        """Return list of coordinate arrays from 2D geometric path."""
        coordinates = []
        points: list[list[float]] = []
        path: list[float] = multi_coordinates.tolist()
        n = 0
        m = 0
        while n < len(path):
            op = int(path[n])
            if op == 0:
                # MOVETO
                if n > 0:
                    coordinates.append(
                        numpy.array(points, dtype=numpy.float32)
                    )
                    points = []
                points.append([path[n + 1], path[n + 2]])
                m = len(points) - 1
                n += 3
            elif op == 1:
                # LINETO
                points.append([path[n + 1], path[n + 2]])
                n += 3
            elif op == 4:
                # CLOSE
                points.append(points[m])
                n += 1
            elif op == 2 or op == 3:
                # QUADTO or CUBICTO
                raise NotImplementedError(
                    f'PathIterator command {op!r} not supported'
                )
            else:
                raise RuntimeError(f'invalid PathIterator command {op!r}')

        coordinates.append(numpy.array(points, dtype=numpy.float32))
        return coordinates

    @staticmethod
    def min_int_coord(value: int | None = None) -> int:
        """Return minimum integer coordinate value.

        The default, -5000, is used by ImageJ.
        A value of -32768 means to use int16 range, 0 means uint16 range.

        """
        if value is None:
            return -5000
        if -32768 <= value <= 0:
            return int(value)
        raise ValueError(f'{value=} out of range')

    @property
    def composite(self) -> bool:
        return self.shape_roi_size > 0

    @property
    def subpixelresolution(self) -> bool:
        return self.version >= 222 and bool(
            self.options & ROI_OPTIONS.SUB_PIXEL_RESOLUTION
        )

    @property
    def drawoffset(self) -> bool:
        return self.subpixelresolution and bool(
            self.options & ROI_OPTIONS.DRAW_OFFSET
        )

    @property
    def subpixelrect(self) -> bool:
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
    def utf16(self) -> str:
        """UTF-16 codec depending on byteorder."""
        return 'utf-16' + ('be' if self.byteorder == '>' else 'le')

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
                value = repr(value).replace('    ', ' ')
                value = value.replace('([[', '([\n    [')
                info.append(f'{name}=numpy.{value},')
            elif value == getattr(ImagejRoi, name):
                pass
            elif isinstance(value, enum.Enum):
                info.append(f'{name}={enumstr(value)},')
            else:
                info.append(f'{name}={value!r},')
        return indent(*info, end='\n)')


def scale_text(text: Any, width: float) -> None:
    """Scale matplotlib text to width in data coordinates."""
    from matplotlib.patheffects import AbstractPathEffect
    from matplotlib.transforms import Bbox

    class TextScaler(AbstractPathEffect):
        def __init__(self, text, width):
            self._text = text
            self._width = width

        def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
            ax = self._text.axes
            renderer = ax.get_figure().canvas.get_renderer()
            bbox = text.get_window_extent(renderer=renderer)
            bbox = Bbox(ax.transData.inverted().transform(bbox))
            if self._width > 0 and bbox.width > 0:
                scale = self._width / bbox.width
                affine = affine.from_values(scale, 0, 0, scale, 0, 0) + affine
            renderer.draw_path(gc, tpath, affine, rgbFace)

    text.set_path_effects([TextScaler(text, width)])


def oval(rect: ArrayLike, /, points: int = 33) -> NDArray[numpy.float32]:
    """Return coordinates of oval from rectangle corners."""
    arr = numpy.array(rect, dtype=numpy.float32)
    c = numpy.linspace(0.0, 2.0 * numpy.pi, num=points, dtype=numpy.float32)
    c = numpy.array([numpy.cos(c), numpy.sin(c)]).T
    r = arr[1] - arr[0]
    r /= 2.0
    c *= r
    c += arr[0] + r
    return c


def indent(*args: Any, sep='', end='') -> str:
    """Return joined string representations of objects with indented lines."""
    text = (sep + '\n').join(
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
    """Return IntEnum or IntFlag as str."""
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
    """Return logging.getLogger('roifile')."""
    return logging.getLogger(__name__.replace('roifile.roifile', 'roifile'))


def test(verbose: bool = False) -> None:
    """Test roifile.ImagejRoi class."""
    # test ROIs from a ZIP file
    rois = ImagejRoi.fromfile('tests/ijzip.zip')
    assert isinstance(rois, list)
    assert len(rois) == 7
    for roi in rois:
        assert roi == ImagejRoi.frombytes(roi.tobytes())
        roi.coordinates()
        if verbose:
            print(roi)
        str(roi)

    # re-write ROIs to a ZIP file
    try:
        os.remove('_test.zip')
    except OSError:
        pass

    def roi_iter():
        # issue #9
        yield from rois

    roiwrite('_test.zip', roi_iter())
    assert roiread('_test.zip') == rois

    # verify box_combined
    rois = roiread('tests/box_combined.roi')
    assert isinstance(rois, ImagejRoi)
    roi = rois
    if verbose:
        print(roi)
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.name == '0464-0752'
    assert roi.roitype == ROI_TYPE.RECT
    assert roi.version == 227
    assert (roi.top, roi.left, roi.bottom, roi.right) == (316, 692, 612, 812)
    coords = roi.coordinates(multi=True)
    assert len(coords) == 31
    assert coords[0][0][0] == 767.0
    assert coords[-1][-1][-1] == 587.0
    assert roi.multi_coordinates is not None
    assert roi.multi_coordinates[0] == 0.0
    with open('tests/box_combined.roi', 'rb') as fh:
        expected = fh.read()
    assert roi.tobytes() == expected
    str(roi)

    roi = ImagejRoi.frompoints([[1, 2], [3, 4], [5, 6]])
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.left == 1
    assert roi.top == 2
    assert roi.right == 6
    assert roi.bottom == 7

    roi = ImagejRoi.frompoints([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.left == 1
    assert roi.top == 2
    assert roi.right == 7
    assert roi.bottom == 8

    roi = ImagejRoi.frompoints([[-5000, 60535], [60534, 65534]])
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.left == -5000, roi.left
    assert roi.top == 60535, roi.top
    assert roi.right == 60535, roi.right
    assert roi.bottom == 65535, roi.bottom

    # issue #7
    roi = ImagejRoi.frompoints(
        numpy.load('tests/issue7.npy').astype(numpy.float32)
    )
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.left == 28357, roi.left
    assert roi.top == 42200, roi.top  # not -23336
    assert roi.right == 28453, roi.right
    assert roi.bottom == 42284, roi.bottom  # not -23252
    coords = roi.coordinates()
    assert roi.integer_coordinates is not None
    assert roi.subpixel_coordinates is not None
    assert roi.integer_coordinates[0, 0] == 0
    assert roi.integer_coordinates[0, 1] == 15
    assert roi.subpixel_coordinates[0, 0] == 28357.0
    assert roi.subpixel_coordinates[0, 1] == 42215.0

    # rotated text
    rois = roiread('tests/text_rotated.roi')
    assert isinstance(rois, ImagejRoi)
    roi = rois
    if verbose:
        print(roi)
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.name == 'Rotated'
    assert roi.roitype == ROI_TYPE.RECT
    assert roi.subtype == ROI_SUBTYPE.TEXT
    assert roi.version == 228
    assert (roi.top, roi.left, roi.bottom, roi.right) == (252, 333, 280, 438)
    assert roi.stroke_color == b'\xff\x00\x00\xff'
    assert roi.text_size == 20
    assert roi.text_justification == 1
    assert roi.text_name == 'SansSerif'
    assert roi.text == 'Enter text...\n'
    with open('tests/text_rotated.roi', 'rb') as fh:
        expected = fh.read()
    assert roi.tobytes() == expected
    str(roi)

    # read a ROI from a TIFF file
    rois = roiread('tests/IJMetadata.tif')
    assert isinstance(rois, list)
    for roi in rois:
        assert roi == ImagejRoi.frombytes(roi.tobytes())
        roi.coordinates()
        if verbose:
            print(roi)
        str(roi)

    assert ImagejRoi() == ImagejRoi()


def main(argv: list[str] | None = None) -> int:
    """Roifile command line usage main function.

    Show all ImageJ ROIs in file or all files in directory::

        python -m roifile file_or_directory

    """
    from glob import glob

    if argv is None:
        argv = sys.argv

    if len(argv) > 1 and '--test' in argv:
        if os.path.exists('../tests'):
            os.chdir('../')
        import doctest

        m: Any
        try:
            import roifile.roifile

            m = roifile.roifile
        except ImportError:
            m = None
        if os.path.exists('tests'):
            print('running tests')
            test()
        print('running doctests')
        doctest.testmod(m)
        return 0

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

    for fname in files:
        print(fname)
        try:
            rois = ImagejRoi.fromfile(fname)
            title = os.path.split(fname)[-1]
            if isinstance(rois, list):
                for roi in rois:
                    print(roi)
                    print()
                    if sys.flags.dev_mode:
                        assert roi == ImagejRoi.frombytes(roi.tobytes())
                if rois:
                    rois[0].plot(rois=rois, title=title)
            else:
                print(rois)
                print()
                if sys.flags.dev_mode:
                    assert rois == ImagejRoi.frombytes(rois.tobytes())
                rois.plot(title=title)
        except ValueError as exc:
            if sys.flags.dev_mode:
                raise
            print(fname, exc)
            continue
    return 0


if __name__ == '__main__':
    sys.exit(main())
