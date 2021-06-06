# roifile.py

# Copyright (c) 2020-2021, Christoph Gohlke
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

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: BSD 3-Clause

:Version: 2021.6.6

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.7.9, 3.8.10, 3.9.5 64-bit <https://www.python.org>`_
* `Numpy 1.20.3 <https://pypi.org/project/numpy/>`_
* `Tifffile 2021.4.8 <https://pypi.org/project/tifffile/>`_  (optional)
* `Matplotlib 3.4.2 <https://pypi.org/project/matplotlib/>`_  (optional)

Revisions
---------
2021.6.6
    Add enums for point types and sizes.
2020.11.28
    Support group attribute.
    Add roiread and roiwrite functions (#3).
    Use UUID as default name of ROI in ImagejRoi.frompoints (#2).
2020.8.13
    Support writing to ZIP file.
    Support os.PathLike file names.
2020.5.28
    Fix int32 to hex color conversion.
    Fix coordinates of closing path.
    Fix reading TIFF files with no overlays.
2020.5.1
    Split positions from counters.
2020.2.12
    Initial release.

Notes
-----
The ImageJ ROI format cannot store integer coordinate values outside the
range of -32768 to 32767 (16-bit signed).

Other Python packages handling ImageJ ROIs:

* `ijpython_roi <https://github.com/dwaithe/ijpython_roi>`_
* `read-roi <https://github.com/hadim/read-roi/>`_

Examples
--------
Create a new ImagejRoi instance from an array of x, y coordinates:

>>> roi = ImagejRoi.frompoints([[1.1, 2.2], [3.3, 4.4], [5.4, 6.6]])
>>> roi.coordinates()
array([[1.1, 2.2],
       [3.3, 4.4],
       [5.4, 6.6]], dtype=float32)
>>> roi.left, roi.left, roi.right, roi.bottom
(1, 1, 5, 6)

Export the instance to an ImageJ ROI formatted byte string or file:

>>> out = roi.tobytes()
>>> out[:4]
b'Iout'
>>> roi.tofile('_test.roi')

Read the ImageJ ROI from the file:

>>> roi2 = ImagejRoi.fromfile('_test.roi')
>>> roi2 == roi
True

Plot the ROI using matplotlib:

>>> roi.plot()

To view the overlays stored in a ROI, ZIP, or TIFF file from a command line,
run ``python -m roifile _test.roi``.

"""

__version__ = '2021.6.6'

__all__ = (
    'roiread',
    'roiwrite',
    'ImagejRoi',
    'ROI_TYPE',
    'ROI_SUBTYPE',
    'ROI_OPTIONS',
    'ROI_POINT_TYPE',
    'ROI_POINT_SIZE',
)

import enum
import os
import struct
import sys
import zipfile

import numpy


def roiread(filename):
    """Return ImagejRoi instance(s) from ROI, ZIP, or TIFF file.

    For ZIP or TIFF files, return a list of ImagejRoi.

    """
    return ImagejRoi.fromfile(filename)


def roiwrite(filename, roi, name=None, mode=None):
    """Write ImagejRoi instance(s) to ROI or ZIP file.

    Write an ImagejRoi instance to a ROI file or write a sequence of ImagejRoi
    instances to a ZIP file. Existing ZIP files are opened for append.

    """
    filename = os.fspath(filename)

    if isinstance(roi, ImagejRoi):
        return roi.tofile(filename, name=name, mode=mode)

    if mode is None:
        mode = 'a' if os.path.exists(filename) else 'w'

    if name is None:
        name = [r.name if r.name else r.autoname for r in roi]
    name = [n if n[-4:].lower() == '.roi' else n + '.roi' for n in name]

    with zipfile.ZipFile(filename, mode) as zf:
        for n, r in zip(name, roi):
            with zf.open(n, 'w') as fh:
                fh.write(r.tobytes())


class ROI_TYPE(enum.IntEnum):
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
    UNDEFINED = 0
    TEXT = 1
    ARROW = 2
    ELLIPSE = 3
    IMAGE = 4
    ROTATED_RECT = 5


class ROI_OPTIONS(enum.IntFlag):
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
    HYBRID = 0
    CROSS = 1
    # CROSSHAIR = 1
    DOT = 2
    CIRCLE = 3


class ROI_POINT_SIZE(enum.IntEnum):
    TINY = 1
    SMALL = 3
    MEDIUM = 5
    LARGE = 7
    EXTRA_LARGE = 11
    XXL = 17
    XXXL = 25


ROI_COLOR_NONE = b'\x00\x00\x00\x00'


class ImagejRoi:
    """Read and write ImageJ ROI format."""

    byteorder = '>'
    roitype = ROI_TYPE(0)
    subtype = ROI_SUBTYPE(0)
    options = ROI_OPTIONS(0)
    name = ''
    props = ''
    version = 217
    top = 0
    left = 0
    bottom = 0
    right = 0
    n_coordinates = 0
    stroke_width = 0
    shape_roi_size = 0
    stroke_color = ROI_COLOR_NONE
    fill_color = ROI_COLOR_NONE
    arrow_style_or_aspect_ratio = 0
    arrow_head_size = 0
    rounded_rect_arc_size = 0
    position = 0
    c_position = 0
    z_position = 0
    t_position = 0
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    xd = 0.0
    yd = 0.0
    widthd = 0.0
    heightd = 0.0
    overlay_label_color = ROI_COLOR_NONE
    overlay_font_size = 0
    group = 0
    image_opacity = 0
    image_size = 0
    float_stroke_width = 0.0
    integer_coordinates = None
    subpixel_coordinates = None
    multi_coordinates = None
    counters = None
    counter_positions = None  # flat indices into TZC array
    text_size = 0
    text_style = 0
    text_justification = 0
    text_name = ''
    text = ''

    @classmethod
    def frompoints(
        cls,
        points=None,
        name=None,
        position=None,
        index=None,
        c=None,
        z=None,
        t=None,
    ):
        """Return ImagejRoi instance from sequence of Point coordinates."""
        if points is None:
            return None

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
                import uuid

                name = str(uuid.uuid1())
            else:
                name = f'F{self.t_position:02}-C{index}'
        self.name = name

        points = numpy.array(points, copy=True)
        if points.dtype.kind == 'f':
            points = numpy.array(points, dtype='f4')
            self.options |= ROI_OPTIONS.SUB_PIXEL_RESOLUTION
        else:
            points = numpy.array(points, dtype='i4')

        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)

        if self.subpixelresolution:
            self.integer_coordinates = numpy.array(
                points - [int(left), int(top)], dtype='i2'
            )
            self.subpixel_coordinates = points
        else:
            points -= [int(left), int(top)]
            self.integer_coordinates = points

        self.n_coordinates = len(self.integer_coordinates)
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)

        return self

    @classmethod
    def fromfile(cls, filename):
        """Return ImagejRoi instance from ROI, ZIP, or TIFF file.

        For ZIP or TIFF files, return a list of ImagejRoi.

        """
        filename = os.fspath(filename)
        if filename[-4:].lower() == '.tif':
            import tifffile

            with tifffile.TiffFile(filename) as tif:
                if not tif.is_imagej:
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
                return [cls.frombytes(roi) for roi in rois]

        if filename[-4:].lower() == '.zip':
            with zipfile.ZipFile(filename) as zf:
                return [
                    cls.frombytes(zf.open(name).read())
                    for name in zf.namelist()
                ]

        with open(filename, 'rb') as fh:
            data = fh.read()
        return cls.frombytes(data)

    @classmethod
    def frombytes(cls, data):
        """Return ImagejRoi instance from bytes."""
        if data[:4] != b'Iout':
            raise ValueError('not an ImageJ ROI')

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
            and (
                self.subtype == ROI_SUBTYPE.ELLIPSE
                or self.subtype == ROI_SUBTYPE.ROTATED_RECT
            )
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
                counters = numpy.ndarray(
                    shape=self.n_coordinates,
                    dtype=self.byteorder + 'u4',
                    buffer=data,
                    offset=counters_offset,
                )
                self.counters = (counters & 0xFF).astype('u1')
                self.counter_positions = counters >> 8

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
            ).copy()

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
            log_warning(f'cannot handle ImagejRoi type {self.roitype!r}')

        return self

    def tofile(self, filename, name=None, mode=None):
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
            with zipfile.ZipFile(filename, mode) as zf:
                with zf.open(name, 'w') as fh:
                    fh.write(self.tobytes())
        else:
            with open(filename, 'wb') as fh:
                fh.write(self.tobytes())

    def tobytes(self):
        """Return ImagejRoi as bytes."""
        result = [b'Iout']

        result.append(
            struct.pack(
                self.byteorder + 'hBxhhhhH',
                self.version,
                self.roitype.value,
                self.top,
                self.left,
                self.bottom,
                self.right,
                self.n_coordinates if self.n_coordinates < 2 ** 16 else 0,
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
            and (
                self.subtype == ROI_SUBTYPE.ELLIPSE
                or self.subtype == ROI_SUBTYPE.ROTATED_RECT
            )
        ):
            result.append(
                struct.pack(
                    self.byteorder + 'ffff', self.x1, self.y1, self.x2, self.y2
                )
            )
        elif self.n_coordinates >= 2 ** 16:
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
            extradata += b'\x00' * 4  # ?

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
                coord = self.integer_coordinates.astype(self.byteorder + 'i2')
                extradata = coord.tobytes(order='F')
            if self.subpixel_coordinates is not None:
                coord = self.subpixel_coordinates.astype(self.byteorder + 'f4')
                extradata += coord.tobytes(order='F')

        elif self.composite and self.roitype == ROI_TYPE.RECT:
            coord = self.multi_coordinates.astype(self.byteorder + 'f4')
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
                counters = counters.astype(self.byteorder + 'u4')
            result.append(counters.tobytes())

        return b''.join(result)

    def plot(self, ax=None, rois=None, title=None, bounds=False, **kwargs):
        """Plot coordinates using matplotlib."""
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
        else:
            fig = None

        if rois is not None:
            for roi in rois:
                roi.plot(ax=ax, **kwargs)
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
            elif self.stroke_width > 0.0:
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
            point = self.coordinates()[0]
            if 'fontsize' not in kwargs and self.text_size > 0:
                kwargs['fontsize'] = self.text_size
            ax.text(point[0], point[1], self.text, **kwargs)
        else:
            for coords in self.coordinates(multi=True):
                ax.plot(coords[:, 0], coords[:, 1], **kwargs)

        ax.plot(self.left, self.bottom, '')
        ax.plot(self.right, self.top, '')

        if fig is not None:
            pyplot.show()

    def coordinates(self, multi=False):
        """Return x, y coordinates as numpy array for display."""
        if self.subpixel_coordinates is not None:
            coords = self.subpixel_coordinates.copy()
        elif self.integer_coordinates is not None:
            coords = self.integer_coordinates + [self.left, self.top]
        elif self.multi_coordinates is not None:
            coords = self.path2coords(self.multi_coordinates)
            if not multi:
                coords = sorted(coords, key=lambda x: x.size)[-1]
            multi = False
        elif self.roitype == ROI_TYPE.LINE:
            coords = numpy.array(
                [[self.x1, self.y1], [self.x2, self.y2]], 'f4'
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
                'f4',
            )
        else:
            coords = numpy.empty((0, 2), dtype=self.byteorder + 'i4')
        return [coords] if multi else coords

    def hexcolor(self, b, default=None):
        """Return color (bytes) as hex triplet or None if black."""
        if b == ROI_COLOR_NONE:
            return default
        if self.byteorder == '>':
            return f'#{b[1]:02x}{b[2]:02x}{b[3]:02x}'
        return f'#{b[3]:02x}{b[2]:02x}{b[1]:02x}'

    @staticmethod
    def path2coords(path):
        """Return list of coordinate arrays from 2D geometric path."""
        coordinates = []
        points = []
        n = 0
        m = 0
        while n < len(path):
            op = int(path[n])
            if op == 0:
                # MOVETO
                if n > 0:
                    coordinates.append(numpy.array(points))
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

        coordinates.append(numpy.array(points))
        return coordinates

    @property
    def composite(self):
        return self.shape_roi_size > 0

    @property
    def subpixelresolution(self):
        return (
            self.version >= 222
            and self.options & ROI_OPTIONS.SUB_PIXEL_RESOLUTION
        )

    @property
    def drawoffset(self):
        return (
            self.subpixelresolution and self.options & ROI_OPTIONS.DRAW_OFFSET
        )

    @property
    def subpixelrect(self):
        return (
            self.version >= 223
            and self.subpixelresolution
            and (
                self.roitype == ROI_TYPE.RECT or self.roitype == ROI_TYPE.OVAL
            )
        )

    @property
    def autoname(self):
        """Return name generated from positions."""
        y = (self.bottom - self.top) // 2
        x = (self.right - self.left) // 2
        name = f'{y:05}-{x:05}'
        if self.counter_positions is not None:
            tzc = int(self.counter_positions.max())
            name = f'{tzc:05}-' + name
        return name

    @property
    def utf16(self):
        """Return UTF-16 codec depending on byteorder."""
        return 'utf-16' + ('be' if self.byteorder == '>' else 'le')

    def __eq__(self, other):
        """Return True if two ImagejRoi are the same."""
        try:
            return self.tobytes() == other.tobytes()
        except Exception:
            return False

    def __str__(self):
        """Return string with information about ImagejRoi."""
        return '\n'.join(
            f' {name} = {value!r}' for name, value in self.__dict__.items()
        )


def oval(rect, points=33, dtype='float32'):
    """Return coordinates of oval from rectangle corners."""
    rect = numpy.array(rect, dtype='float32')
    c = numpy.linspace(0.0, 2.0 * numpy.pi, num=points, dtype='float32')
    c = numpy.array([numpy.cos(c), numpy.sin(c)]).T
    r = rect[1] - rect[0]
    r /= 2.0
    c *= r
    c += rect[0] + r
    return c.astype(dtype)


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)


def test_imagejroi():
    """Test roifile.ImagejRoi class."""
    # test ROIs from a ZIP file
    rois = ImagejRoi.fromfile('tests/ijzip.zip')
    assert len(rois) == 7
    for roi in rois:
        assert roi.tobytes() == ImagejRoi.frombytes(roi.tobytes()).tobytes()
        roi.coordinates()
        roi.__str__()

    # re-write ROIs to a ZIP file
    try:
        os.remove('_test.zip')
    except OSError:
        pass
    roiwrite('_test.zip', rois)
    assert roiread('_test.zip') == rois

    # verify box_combined
    roi = roiread('tests/box_combined.roi')
    assert roi.tobytes() == ImagejRoi.frombytes(roi.tobytes()).tobytes()
    assert roi.name == '0464-0752'
    assert roi.roitype == ROI_TYPE.RECT
    assert roi.version == 227
    assert (roi.top, roi.left, roi.bottom, roi.right) == (316, 692, 612, 812)
    coords = roi.coordinates(multi=True)
    assert len(coords) == 31
    assert coords[0][0][0] == 767.0
    assert coords[-1][-1][-1] == 587.0
    assert roi.multi_coordinates[0] == 0.0
    roi.__str__()

    # read a ROI from a TIFF file
    for roi in roiread('tests/IJMetadata.tif'):
        assert roi == ImagejRoi.frombytes(roi.tobytes())
        roi.coordinates()
        roi.__str__()

    assert ImagejRoi() == ImagejRoi()


def main(argv=None, test=False):
    """Roifile command line usage main function.

    Show all ImageJ ROIs in file or all files in directory:

    ``python -m roifile file_or_directory``

    """
    from glob import glob

    if argv is None:
        argv = sys.argv

    if len(argv) > 1 and '--test' in argv:
        if os.path.exists('../tests'):
            os.chdir('../')
        import doctest

        try:
            import roifile.roifile as m
        except ImportError:
            m = None
        if os.path.exists('tests'):
            print('running tests')
            test_imagejroi()
        print('running doctests')
        doctest.testmod(m)
        return

    if len(argv) == 1:
        files = glob('*.roi')
    elif '*' in argv[1]:
        files = glob(argv[1])
    elif os.path.isdir(argv[1]):
        files = glob(f'{argv[1]}/*.roi')
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
                if rois:
                    rois[0].plot(rois=rois, title=title)
            else:
                print(rois)
                print()
                rois.plot(title=title)
        except ValueError as exc:
            # raise  # enable for debugging
            print(fname, exc)
            continue


if __name__ == '__main__':
    sys.exit(main())
