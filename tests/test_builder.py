import pathlib

import numpy as np
import pytest
from astropy.io import fits

from regularizepsf.builder import ArrayPSFBuilder, _average_patches, _find_patches
from regularizepsf.psf import ArrayPSF

TEST_DIR = pathlib.Path(__file__).parent.resolve()

def test_find_patches():
    img_path = str(TEST_DIR / "data/compressed_dash.fits")
    image_array = fits.getdata(img_path).astype(float)
    patches = _find_patches(image_array, 3, None, 1, 32, 50)
    for coord, patch in patches.items():
        assert coord[0] == 50
        print(coord)
        assert patch.shape == (32, 32)

def test_averaging():
    collection = {(0, 0, 0): np.full((10, 10), .3),
             (1, 0, 0): np.full((10, 10), .5),
             (2, 0, 0): np.full((10, 10), .9),
             # Exercise the nan-rejection in CoordinatePatchCollection.average()
             (3, 0, 0): np.full((10, 10), np.nan),
             }
    for patch in collection.values():
        # Make the normalization of each patch a no-op
        patch[5, 5] = 1

    averaged_collection, counts = _average_patches(collection, np.array([[0, 0]]),  method='median')
    expected = np.nanmedian([.3, .5, .9])
    assert averaged_collection[(0, 0)][1, 1] == expected

    averaged_collection, counts = _average_patches(collection, np.array([[0, 0]]), method='mean')
    expected = np.nanmean([.3, .5, .9])
    assert averaged_collection[(0, 0)][1, 1] == expected

    averaged_collection, counts = _average_patches(collection, np.array([[0, 0]]), method='percentile', percentile=20)
    expected = np.nanpercentile([.3, .5, .9], 20)
    assert averaged_collection[(0, 0)][1, 1] == expected


@pytest.mark.parametrize("method", ["mean", "median", "percentile"])
def test_find_stars_and_average_path(method):
    img_path = str(TEST_DIR / "data/compressed_dash.fits")
    builder = ArrayPSFBuilder(32)
    example, _ = builder.build([img_path], average_method=method, hdu_choice=1)
    assert isinstance(example, ArrayPSF)
    assert example.sample_shape == (32, 32)

@pytest.mark.parametrize("method", ["mean", "median", "percentile"])
def test_find_stars_and_average_array(method):
    img_path = str(TEST_DIR / "data/compressed_dash.fits")
    image_array = fits.getdata(img_path).astype(float)
    image_array = image_array.reshape((1, *image_array.shape))

    # Use a mask to only process part of the image, to speed up this test
    mask = np.ones_like(image_array, dtype=bool)
    mask[:, :800, :800] = 0

    builder = ArrayPSFBuilder(32)
    example, _ = builder.build(image_array, mask, average_method=method)
    assert isinstance(example, ArrayPSF)
    assert example.sample_shape == (32, 32)


@pytest.mark.parametrize("method", ["mean", "median", "percentile"])
def test_find_stars_and_average_generator(method):
    img_path = str(TEST_DIR / "data/compressed_dash.fits")
    image_array = fits.getdata(img_path).astype(float)
    image_array = image_array.reshape((1, *image_array.shape))
    def generator():
        yield image_array[0]

    builder = ArrayPSFBuilder(32)
    example, _ = builder.build(generator(), average_method=method)
    assert isinstance(example, ArrayPSF)
    assert example.sample_shape == (32, 32)
