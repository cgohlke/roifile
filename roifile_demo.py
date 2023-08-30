# roifile_demo.py

"""Demonstrate the roifile module.

Use the roifile and tifffile modules to read and write the results of image
segmentation from/to ImageJ TIFF files.

"""

import numpy
from matplotlib import pyplot
from skimage.measure import find_contours, label, regionprops
from tifffile import TiffFile, imwrite

from roifile import ImagejRoi


def plot_image_overlays(image, overlays, **kwargs):
    """Plot image and overlays (bytes) using matplotlib."""
    fig, ax = pyplot.subplots()
    ax.imshow(image, cmap='gray')
    if not isinstance(overlays, list):
        overlays = [overlays]
    for overlay in overlays:
        roi = ImagejRoi.frombytes(overlay)
        roi.plot(ax, **kwargs)
    pyplot.show()


# open an ImageJ TIFF file and read the image and overlay data
# https://github.com/csachs/imagej-tiff-meta/
# blob/b6a74daa8c2adf7023d20a447d9a2799614c857a/box.tif
with TiffFile('tests/box.tif') as tif:
    image = tif.pages[0].asarray()
    assert tif.imagej_metadata is not None
    overlays = tif.imagej_metadata['Overlays']

plot_image_overlays(image, overlays)

# segment the image with scikit-image
labeled = label(image > 0.5 * image.max())
for region in regionprops(labeled):
    if region.area > 10000:
        labeled[labeled == region.label] = 0
    elif region.area < 100:
        labeled[labeled == region.label] = 0
segmentation = 1.0 * (labeled > 0)

# create ImageJ overlays from segmentation results
overlays = [
    ImagejRoi.frompoints(numpy.round(contour)[:, ::-1]).tobytes()
    for contour in find_contours(segmentation, level=0.9999)
]

plot_image_overlays(image, overlays, lw=5)

# write the image and overlays to a new ImageJ TIFF file
imwrite('roi_test.tif', image, imagej=True, metadata={'Overlays': overlays})
