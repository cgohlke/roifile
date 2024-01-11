Read and write ImageJ ROI format
================================

Roifile is a Python library to read, write, create, and plot `ImageJ`_ ROIs,
an undocumented and ImageJ application specific format to store regions of
interest, geometric shapes, paths, text, and whatnot for image overlays.

.. _ImageJ: https://imagej.net

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2024.1.10
:DOI: `10.5281/zenodo.6941603 <https://doi.org/10.5281/zenodo.6941603>`_

Quickstart
----------

Install the roifile package and all dependencies from the
`Python Package Index <https://pypi.org/project/roifile/>`_::

    python -m pip install -U roifile[all]

View overlays stored in a ROI, ZIP, or TIFF file::

    python -m roifile file.roi

See `Examples`_ for using the programming interface.

Source code, examples, and support are available on
`GitHub <https://github.com/cgohlke/roifile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.4, 3.12.1
- `Numpy <https://pypi.org/project/numpy/>`_ 1.26.3
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2023.12.9 (optional)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.8.2 (optional)

Revisions
---------

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

- Update metadata.

2022.3.18

- Fix creating ROIs from float coordinates exceeding int16 range (#7).
- Fix bottom-right bounds in ImagejRoi.frompoints.

2022.2.2

- Add type hints.
- Change ImagejRoi to dataclass.
- Drop support for Python 3.7 and numpy < 1.19 (NEP29).

2021.6.6

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
