Read and write ImageJ ROI format
================================

Roifile is a Python library to read, write, create, and plot `ImageJ`_ ROIs,
an undocumented and ImageJ application specific format to store regions of
interest, geometric shapes, paths, text, and whatnot for image overlays.

.. _ImageJ: https://imagej.net

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: BSD 3-Clause

:Version: 2022.2.2

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.8.10, 3.9.10, 3.10.2 64-bit <https://www.python.org>`_
* `Numpy 1.21.5 <https://pypi.org/project/numpy/>`_
* `Tifffile 2021.11.2 <https://pypi.org/project/tifffile/>`_  (optional)
* `Matplotlib 3.4.3 <https://pypi.org/project/matplotlib/>`_  (optional)

Revisions
---------
2022.2.2
    Add type hints.
    Change ImagejRoi to dataclass.
    Drop support for Python 3.7 and numpy < 1.19 (NEP29).
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
