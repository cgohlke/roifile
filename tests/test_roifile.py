# test_roifile.py

# Copyright (c) 2020-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the roifile package.

:Version: 2026.2.10

"""

import contextlib
import glob
import io
import os
import pathlib
import pickle
import sys
from collections.abc import Iterator
from typing import Any

import numpy
import pytest
from matplotlib import pyplot

import roifile
from roifile import (
    ROI_COLOR_NONE,
    ROI_OPTIONS,
    ROI_POINT_SIZE,
    ROI_POINT_TYPE,
    ROI_SUBTYPE,
    ROI_TYPE,
    ImagejRoi,
    __version__,
    roiread,
    roiwrite,
)

HERE = pathlib.Path(os.path.dirname(__file__))
DATA = HERE / 'data'


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert roifile versions match docstrings."""
    ver = ':Version: ' + __version__
    assert ver in __doc__
    assert ver in roifile.__doc__


def test_read_zip_file():
    """Test reading ROIs from a ZIP file."""
    rois: Any = ImagejRoi.fromfile(DATA / 'ijzip.zip')
    assert isinstance(rois, list)
    assert len(rois) == 7
    for roi in rois:
        assert roi == ImagejRoi.frombytes(roi.tobytes())
        roi.coordinates()
        _ = roi.properties
        str(roi)


def test_write_zip_file():
    """Test writing ROIs to a ZIP file."""
    rois: Any = ImagejRoi.fromfile(DATA / 'ijzip.zip')

    with contextlib.suppress(OSError):
        os.remove('_test.zip')

    def roi_iter() -> Iterator[ImagejRoi]:
        yield from rois

    roiwrite('_test.zip', roi_iter())
    assert roiread('_test.zip') == rois


def test_read_tiff_file():
    """Test reading ROIs from a TIFF file."""
    rois = roiread(DATA / 'IJMetadata.tif')
    assert isinstance(rois, list)
    for roi in rois:
        assert roi == ImagejRoi.frombytes(roi.tobytes())
        roi.coordinates()
        str(roi)


def test_empty_roi():
    """Test empty ROI equality."""
    assert ImagejRoi() == ImagejRoi()


def test_box_combined():
    """Test box_combined composite ROI."""
    rois = roiread(DATA / 'box_combined.roi')
    assert isinstance(rois, ImagejRoi)
    roi = rois
    str(roi)
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
    with open(DATA / 'box_combined.roi', 'rb') as fh:
        expected = fh.read()
    assert roi.properties == {}
    assert roi.tobytes() == expected
    str(roi)


def test_frompoints_integer():
    """Test creating ROI from integer coordinates."""
    roi = ImagejRoi.frompoints([[1, 2], [3, 4], [5, 6]])
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.left == 1
    assert roi.top == 2
    assert roi.right == 6
    assert roi.bottom == 7

    # test ROI_POINT_SIZE
    roi.roitype = ROI_TYPE.POINT
    roi.point_size = ROI_POINT_SIZE.SMALL
    assert roi.point_size == ROI_POINT_SIZE.SMALL
    roi.point_size = ROI_POINT_SIZE.LARGE
    assert roi.point_size == ROI_POINT_SIZE.LARGE
    roi.point_size = 17
    assert roi.point_size == ROI_POINT_SIZE.XXL
    roi.point_size = 100
    assert roi.point_size == ROI_POINT_SIZE.UNKNOWN
    assert roi.point_size.value == 100

    # test ROI_POINT_TYPE
    roi.point_type = ROI_POINT_TYPE.CROSS
    assert roi.point_type == ROI_POINT_TYPE.CROSS
    roi.point_type = ROI_POINT_TYPE.DOT
    assert roi.point_type == ROI_POINT_TYPE.DOT
    roi.point_type = 3
    assert roi.point_type == ROI_POINT_TYPE.CIRCLE
    roi.point_type = 100
    assert roi.point_type == ROI_POINT_TYPE.UNKNOWN
    assert roi.point_type.value == 100

    # verify round-trip with point properties
    roi2 = ImagejRoi.frombytes(roi.tobytes())
    assert roi2.point_size == roi.point_size
    assert roi2.point_type == roi.point_type
    assert roi2 == roi


def test_frompoints_float():
    """Test creating ROI from float coordinates."""
    roi = ImagejRoi.frompoints([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.left == 1
    assert roi.top == 2
    assert roi.right == 7
    assert roi.bottom == 8


def test_frompoints_large_coordinates():
    """Test creating ROI from large coordinate values."""
    roi = ImagejRoi.frompoints([[-5000, 60535], [60534, 65534]])
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.left == -5000, roi.left
    assert roi.top == 60535, roi.top
    assert roi.right == 60535, roi.right
    assert roi.bottom == 65535, roi.bottom


@pytest.mark.parametrize(
    'points',
    [
        [[0, 0], [10, 10]],  # small positive coords
        [[-5000, 0], [100, 100]],  # boundary at -5000
        [[0, 0], [60535, 60535]],  # large positive at boundary
        [[-5000, -5000], [60535, 60535]],  # mixed range
        [[0, 0], [32767, 32768]],  # near int16 max boundary
        [[0, 0], [65534, 65535]],  # near uint16 max boundary
        [[-5000, 60535], [60534, 65534]],  # original issue #13
    ],
)
def test_coordinate_wrapping_roundtrip(points):
    """Test that coordinates correctly roundtrip through int16 wrapping."""
    roi1 = ImagejRoi.frompoints(points)
    data = roi1.tobytes()
    roi2 = ImagejRoi.frombytes(data)

    assert roi1 == roi2
    assert numpy.array_equal(
        roi1.integer_coordinates, roi2.integer_coordinates
    )
    assert numpy.array_equal(roi1.coordinates(), roi2.coordinates())


def test_min_int_coord_parameter():
    """Test that min_int_coord parameter is respected."""
    # test with default -5000
    roi = ImagejRoi.frompoints([[-5000, 0], [100, 100]])
    data = roi.tobytes()
    roi_default = ImagejRoi.frombytes(data)
    assert roi_default == roi

    # test with -32768 (full int16 range)
    roi2 = ImagejRoi.frompoints([[-32768, 0], [100, 100]])
    data2 = roi2.tobytes()
    roi_int16 = ImagejRoi.frombytes(data2, min_int_coord=-32768)
    assert roi_int16 == roi2

    # test with 0 (uint16 range, no negative coordinates)
    roi3 = ImagejRoi.frompoints([[0, 0], [60535, 60535]])
    data3 = roi3.tobytes()
    roi_uint16 = ImagejRoi.frombytes(data3, min_int_coord=0)
    assert roi_uint16 == roi3


@pytest.mark.parametrize('kind', ['uint8', 'uint16', 'float32', 'rgb'])
def test_image_subtype(kind):
    """Test creating IMAGE subtype ROI."""
    image: numpy.ndarray
    if kind == 'rgb':
        image = numpy.arange(300, dtype=numpy.uint8).reshape(10, 10, 3)
    else:
        image = numpy.arange(100, dtype=kind).reshape(10, 10)

    roi = ImagejRoi()
    roi.version = 228
    roi.name = 'Image ROI'
    roi.roitype = ROI_TYPE.RECT
    roi.subtype = ROI_SUBTYPE.IMAGE
    roi.left = 50
    roi.top = 100
    roi.image_opacity = 128
    roi.image = image

    with open(f'{kind}.roi', 'wb') as fh:
        fh.write(roi.tobytes())

    assert roi.subtype == ROI_SUBTYPE.IMAGE
    assert roi.image_size > 0
    assert roi.image_data is not None
    assert roi.right == roi.left + image.shape[1]
    assert roi.bottom == roi.top + image.shape[0]

    # test round-trip
    roi2 = ImagejRoi.frombytes(roi.tobytes())
    assert roi2 == roi
    assert roi2.subtype == ROI_SUBTYPE.IMAGE
    assert roi2.image_size == roi.image_size
    assert roi2.image_opacity == 128

    # test that image can be decoded
    decoded_image = roi2.image
    assert decoded_image is not None
    assert decoded_image.shape == image.shape
    numpy.testing.assert_array_equal(decoded_image, image)

    str(roi)
    str(roi2)


def test_rotated_text():
    """Test reading rotated text ROI."""
    rois = roiread(DATA / 'text_rotated.roi')
    assert isinstance(rois, ImagejRoi)
    roi = rois
    str(roi)
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
    with open(DATA / 'text_rotated.roi', 'rb') as fh:
        expected = fh.read()
    assert roi.tobytes() == expected
    str(roi)


def test_properties():
    """Test ROI properties."""
    rois = ImagejRoi.fromfile(DATA / '27197299958_88cf5966d3_b.tif')
    assert isinstance(rois, list)
    assert len(rois) == 8

    for roi in rois:
        roi.props.endswith('\n')
        assert roi == ImagejRoi.frombytes(roi.tobytes())
        roi.coordinates()
        str(roi)

    roi = rois[0]
    assert roi.roitype == ROI_TYPE.LINE
    assert roi.subtype == ROI_SUBTYPE.UNDEFINED
    assert roi.options == (
        ROI_OPTIONS.OVERLAY_LABELS | ROI_OPTIONS.OVERLAY_BACKGROUNDS
    )
    assert roi.version == 228
    assert roi.name == 'Plat'
    assert roi.stroke_color == b'\xff\xff\x00\x00'
    assert roi.fill_color == ROI_COLOR_NONE
    assert len(roi.props) == 418
    assert roi.props.startswith('%Area: 0\nAbutment_implant_misfit: No\n')
    assert roi.props.endswith('X: 48\nY: 98.500\n')

    props = roi.properties
    assert isinstance(props, dict)
    assert len(props) == 26
    assert props['%Area'] == 0
    assert props['Y'] == 98.5
    assert props['Angle'] == -1.193
    assert props['Roi'] == 'Plat'
    assert props['Poor_Quality'] == 'No'

    roi.properties = props  # encode roi.props
    assert roi.properties == props  # decode roi.props
    assert len(roi.props) == 415
    assert roi.props.startswith('%Area: 0\nAbutment_implant_misfit: No\n')
    assert roi.props.endswith('X: 48\nY: 98.5\n')  # float formatting different
    assert roi == ImagejRoi.frombytes(roi.tobytes())

    props['Bool'] = False
    roi.properties = props
    assert roi.properties['Bool'] is False
    assert 'BY: 98\nBool: false\n' in roi.props  # sorted by key


def test_zipfile():
    """Test ROIs from ZIP file."""
    rois = ImagejRoi.fromfile(DATA / 'ijzip.zip')
    assert isinstance(rois, list)
    assert len(rois) == 7
    for roi in rois:
        assert roi == ImagejRoi.frombytes(roi.tobytes())
        roi.coordinates()
        str(roi)


def test_pickle():
    """Test pickling ROI."""
    rois = ImagejRoi.fromfile(DATA / 'ijzip.zip')
    fh = io.BytesIO()
    pickle.dump(rois, fh)
    for roi0, roi1 in zip(
        rois, pickle.loads(fh.getvalue()), strict=True  # noqa: S301
    ):
        assert roi0 == roi1


def test_issue_7():
    """Test issue #7: large coordinate values."""
    roi = ImagejRoi.frompoints(
        numpy.load(DATA / 'issue7.npy').astype(numpy.float32)
    )
    assert roi == ImagejRoi.frombytes(roi.tobytes())
    assert roi.left == 28357, roi.left
    assert roi.top == 42200, roi.top  # not -23336
    assert roi.right == 28453, roi.right
    assert roi.bottom == 42284, roi.bottom  # not -23252
    _ = roi.coordinates()
    assert roi.integer_coordinates is not None
    assert roi.subpixel_coordinates is not None
    assert roi.integer_coordinates[0, 0] == 0
    assert roi.integer_coordinates[0, 1] == 15
    assert roi.subpixel_coordinates[0, 0] == 28357.0
    assert roi.subpixel_coordinates[0, 1] == 42215.0


def test_issue_9():
    """Test issue #9: roiwrite with iterable input."""
    # re-write ROIs to a ZIP file
    rois: Any = ImagejRoi.fromfile(DATA / 'ijzip.zip')
    assert len(rois) == 7

    # write list
    with contextlib.suppress(OSError):
        os.remove('_test.zip')
    roiwrite('_test.zip', rois)
    assert roiread('_test.zip') == rois

    # provide names
    with contextlib.suppress(OSError):
        os.remove('_test.zip')
    roiwrite('_test.zip', rois, name=[str(i) for i in range(len(rois))])
    assert roiread('_test.zip') == rois

    # write generator, issue #9
    def roi_iter() -> Iterator[ImagejRoi]:
        yield from rois

    with contextlib.suppress(OSError):
        os.remove('_test.zip')
    roiwrite('_test.zip', roi_iter())
    assert roiread('_test.zip') == rois


@pytest.mark.parametrize(
    'filename', glob.glob('*.roi', root_dir=DATA, recursive=False)
)
def test_glob_roi(filename):
    """Test read all ROI files."""
    if 'defective' in filename:
        pytest.xfail(reason='file is marked defective')
    filename = DATA / filename
    roi = ImagejRoi.fromfile(filename)
    assert isinstance(roi, ImagejRoi)
    str(roi)
    roi.plot(show=False)
    pyplot.close()


@pytest.mark.parametrize(
    'filename', glob.glob('*.zip', root_dir=DATA, recursive=False)
)
def test_glob_zip(filename):
    """Test read all ZIP files."""
    if 'defective' in filename:
        pytest.xfail(reason='file is marked defective')
    filename = DATA / filename
    rois = roiread(filename)
    assert isinstance(rois, list)
    for roi in rois:
        str(roi)
        roi.plot(show=False)
    pyplot.close()


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=roifile')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
