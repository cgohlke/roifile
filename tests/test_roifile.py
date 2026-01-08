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

:Version: 2026.1.8

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


def test_roifile():
    """Test ImagejRoi class."""
    # TODO: split into smaller test functions

    # test ROIs from a ZIP file
    rois: Any = ImagejRoi.fromfile(DATA / 'ijzip.zip')
    assert isinstance(rois, list)
    assert len(rois) == 7
    for roi in rois:
        assert roi == ImagejRoi.frombytes(roi.tobytes())
        roi.coordinates()
        str(roi)

    # re-write ROIs to a ZIP file
    with contextlib.suppress(OSError):
        os.remove('_test.zip')

    def roi_iter() -> Iterator[ImagejRoi]:
        # issue #9
        yield from rois

    roiwrite('_test.zip', roi_iter())
    assert roiread('_test.zip') == rois

    # verify box_combined
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
        numpy.load(DATA / 'issue7.npy').astype(numpy.float32)
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

    # read a ROI from a TIFF file
    rois = roiread(DATA / 'IJMetadata.tif')
    assert isinstance(rois, list)
    for roi in rois:
        assert roi == ImagejRoi.frombytes(roi.tobytes())
        roi.coordinates()
        str(roi)

    assert ImagejRoi() == ImagejRoi()


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
    'fname', glob.glob('*.roi', root_dir=DATA, recursive=False)
)
def test_glob_roi(fname):
    """Test read all ROI files."""
    if 'defective' in fname:
        pytest.xfail(reason='file is marked defective')
    fname = DATA / fname
    roi = ImagejRoi.fromfile(fname)
    assert isinstance(roi, ImagejRoi)
    str(roi)
    roi.plot(show=False)
    pyplot.close()


@pytest.mark.parametrize(
    'fname', glob.glob('*.zip', root_dir=DATA, recursive=False)
)
def test_glob_zip(fname):
    """Test read all ZIP files."""
    if 'defective' in fname:
        pytest.xfail(reason='file is marked defective')
    fname = DATA / fname
    rois = roiread(fname)
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
