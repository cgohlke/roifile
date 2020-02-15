Read and write ImageJ ROI format
================================

Roifile is a Python library to read, write, create, and plot `ImageJ`_ ROIs,
an undocumented and ImageJ application specific format
to store regions of interest, geometric shapes, paths, text, and whatnot
for image overlays.

.. _ImageJ: https://imagej.net

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.2.12

Requirements
------------
* `CPython >= 3.6 <https://www.python.org>`_
* `Numpy 1.15.1 <https://www.numpy.org>`_
* `Tifffile 2019.7.2 <https://pypi.org/project/tifffile/>`_  (optional)
* `Matplotlib 3.1 <https://pypi.org/project/matplotlib/>`_  (optional)

Revisions
---------
2020.2.12
    Initial release.

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
