import os.path
import pathlib

from astropy.io import fits
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

from regularizepsf.fitter import CoordinatePatchCollection, CoordinateIdentifier
from regularizepsf.exceptions import InvalidSizeError


TEST_DIR = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def incrementing_5_image_set_100x100():
    """ 5 images of size 100x100 that increment value from 0 to 4"""
    return np.stack([np.zeros((100, 100))+i for i in range(5)])


def test_coordinate_patch_collection_extraction_empty_coordinates(incrementing_5_image_set_100x100):
    cpc = CoordinatePatchCollection.extract(incrementing_5_image_set_100x100, [], 10)
    assert len(cpc) == 0
    assert cpc.size is None


def test_coordinate_patch_collection_extraction_one_coordinate(incrementing_5_image_set_100x100):
    cpc = CoordinatePatchCollection.extract(incrementing_5_image_set_100x100, [CoordinateIdentifier(0, 0, 0)], 10)
    assert len(cpc) == 1
    assert cpc.size == 10
    assert CoordinateIdentifier(0, 0, 0) in cpc
    assert np.all(cpc[CoordinateIdentifier(0, 0, 0)] == np.zeros((10, 10)))


@st.composite
def list_of_coordinate_identifiers(draw):
    n = draw(st.integers(min_value=1, max_value=50))
    i = st.lists(st.integers(0, 4), min_size=n, max_size=n)
    x = st.lists(st.integers(0, 89), min_size=n, max_size=n)
    y = st.lists(st.integers(0, 89), min_size=n, max_size=n)

    return [CoordinateIdentifier(ii, xx, yy) for ii, xx, yy, in zip(draw(i), draw(x), draw(y))]


@given(coords=list_of_coordinate_identifiers())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_coordinate_patch_collection_extraction_many_coordinates(coords, incrementing_5_image_set_100x100):
    cpc = CoordinatePatchCollection.extract(incrementing_5_image_set_100x100, coords, 10)

    num_distinct_coords = len(set(coords))
    assert len(cpc) == num_distinct_coords
    assert cpc.size == 10
    assert len(list(cpc.values())) == num_distinct_coords
    assert len(list(cpc.keys())) == num_distinct_coords
    assert len(list(cpc.items())) == num_distinct_coords


def test_missing_item_retrieval():
    collection = CoordinatePatchCollection({CoordinateIdentifier(0, 0, 0): np.zeros((10, 10))})
    with pytest.raises(IndexError):
        item = collection[CoordinateIdentifier(1, 1, 1)]


def test_saving_and_loading():
    collection = CoordinatePatchCollection({CoordinateIdentifier(0, 0, 0): np.zeros((10, 10))})
    collection.save("test.psf")
    assert os.path.isfile("test.psf")
    loaded = CoordinatePatchCollection.load("test.psf")
    assert isinstance(loaded, CoordinatePatchCollection)
    os.remove("test.psf")


def test_coordinate_patch_average():
    collection = CoordinatePatchCollection(
            {CoordinateIdentifier(0, 0, 0): np.full((10, 10), .3),
             CoordinateIdentifier(1, 0, 0): np.full((10, 10), .5),
             CoordinateIdentifier(2, 0, 0): np.full((10, 10), .9),
             # Exercise the nan-rejection in CoordinatePatchCollection.average()
             CoordinateIdentifier(3, 0, 0): np.full((10, 10), np.nan),
             })
    for patch in collection.values():
        # Make the normalization of each patch a no-op
        patch[5, 5] = 1

    averaged_collection = collection.average(
            np.array([[0, 0]]), 10, 10, mode='median')
    expected = np.nanmedian([.3, .5, .9])
    assert averaged_collection[CoordinateIdentifier(None, 0, 0)][1, 1] == expected

    averaged_collection = collection.average(
            np.array([[0, 0]]), 10, 10, mode='mean')
    expected = np.nanmean([.3, .5, .9])
    assert averaged_collection[CoordinateIdentifier(None, 0, 0)][1, 1] == expected

    averaged_collection = collection.average(
            np.array([[0, 0]]), 10, 10, mode='percentile', percentile=20)
    expected = np.nanpercentile([.3, .5, .9], 20)
    assert averaged_collection[CoordinateIdentifier(None, 0, 0)][1, 1] == expected


def test_calculate_pad_shape():
    collection = CoordinatePatchCollection({CoordinateIdentifier(0, 0, 0): np.zeros((10, 10))})
    assert collection.size == 10
    assert collection._calculate_pad_shape(20) == ((5, 5), (5, 5))


def test_negative_pad_shape_errors():
    collection = CoordinatePatchCollection({CoordinateIdentifier(0, 0, 0): np.zeros((10, 10))})
    with pytest.raises(InvalidSizeError):
        collection._calculate_pad_shape(1)


def test_odd_pad_shape_errors():
    collection = CoordinatePatchCollection({CoordinateIdentifier(0, 0, 0): np.zeros((10, 10))})
    with pytest.raises(InvalidSizeError):
        collection._calculate_pad_shape(11)


def test_validate_average_mode():
    with pytest.raises(ValueError):
        CoordinatePatchCollection._validate_average_mode(
                "nonexistent_method", 1)

    # Ensure valid modes are accepted
    for mode in ('mean', 'median', 'percentile'):
        CoordinatePatchCollection._validate_average_mode(mode, 1)

    # Check invalid percentile values
    with pytest.raises(ValueError):
        CoordinatePatchCollection._validate_average_mode(
                "percentile", -1)
    with pytest.raises(ValueError):
        CoordinatePatchCollection._validate_average_mode(
                "percentile", 101)


def test_find_stars_and_average():
    img_path = str(TEST_DIR / "data/DASH.fits")
    example = CoordinatePatchCollection.find_stars_and_average([img_path], 32, 100)
    assert isinstance(example, CoordinatePatchCollection)
    for loc, patch in example.patches.items():
        assert patch.shape == (100, 100)


def test_find_stars_and_average_powers_of_2():
    img_path = str(TEST_DIR / "data/DASH.fits")
    example = CoordinatePatchCollection.find_stars_and_average([img_path], 32, 128)
    assert isinstance(example, CoordinatePatchCollection)
    for loc, patch in example.patches.items():
        assert patch.shape == (128, 128)


def test_find_stars_and_average_powers_of_2_mean():
    img_path = str(TEST_DIR / "data/DASH.fits")
    example = CoordinatePatchCollection.find_stars_and_average([img_path], 32, 128, average_mode='mean')
    assert isinstance(example, CoordinatePatchCollection)
    for loc, patch in example.patches.items():
        assert patch.shape == (128, 128)


def test_find_stars_and_average_image_formats():
    """Runs find_stars_and_average with the three possible input-data formats"""
    img_paths = [str(TEST_DIR / "data/DASH.fits")]

    imgs_array = fits.getdata(img_paths[0]).astype(float)
    imgs_array = imgs_array.reshape((1, *imgs_array.shape))

    # Use a mask to only process part of the image, to speed up this test
    mask = np.ones_like(imgs_array, dtype=bool)
    mask[:, :800, :800] = 0

    example_ndarray = CoordinatePatchCollection.find_stars_and_average(
            imgs_array, 32, 100, star_mask=mask)

    example_list = CoordinatePatchCollection.find_stars_and_average(
            img_paths, 32, 100, star_mask=mask)

    def generator():
        yield imgs_array[0]
    example_generator = CoordinatePatchCollection.find_stars_and_average(
            generator(), 32, 100, star_mask=mask)

    # Check that we got the correct output type for each one
    for example in (example_list, example_generator, example_ndarray):
        assert isinstance(example, CoordinatePatchCollection)

    # Check that the keys of the patch dictionaries are the same for all three
    assert set(example_list.patches.keys()) == set(example_generator.patches.keys())
    assert set(example_list.patches.keys()) == set(example_ndarray.patches.keys())

    # Check that the patches themselves are the same for all three
    for loc in example_list.patches.keys():
        assert np.all(example_list.patches[loc] == example_generator.patches[loc])
        assert np.all(example_list.patches[loc] == example_ndarray.patches[loc])


def test_find_stars_and_average_mask_formats(tmp_path):
    """ Test star-finding masks, in all accepted formats"""
    img_paths = [str(TEST_DIR / "data/DASH.fits")]
    imgs_array = fits.getdata(img_paths[0])
    # Cut down the data size so this test runs quicker
    imgs_array = imgs_array[:800, :800].astype(float)
    imgs_array = imgs_array.reshape((1, *imgs_array.shape))

    # Mask out everything except one corner
    mask_array = np.ones_like(imgs_array, dtype=bool)
    mask_array[:, :402, :402] = 0

    # Try all the formats in which masks are accepted

    example_ndarray = CoordinatePatchCollection.find_stars_and_average(
            imgs_array, 32, 200, star_mask=mask_array)

    def generator():
        yield mask_array[0]
    example_generator = CoordinatePatchCollection.find_stars_and_average(
            imgs_array, 32, 200, star_mask=generator())

    mask_fname = str(tmp_path / "mask.fits")
    fits.writeto(mask_fname, mask_array[0].astype(int))
    example_file = CoordinatePatchCollection.find_stars_and_average(
            imgs_array, 32, 200, star_mask=[mask_fname])

    example_no_mask = CoordinatePatchCollection.find_stars_and_average(
            imgs_array, 32, 200, star_mask=None)

    for loc in example_file.patches.keys():
        # Check that the patches are the same for all three mask formats
        assert np.all(example_file.patches[loc] == example_generator.patches[loc])
        assert np.all(example_file.patches[loc] == example_ndarray.patches[loc])

        if loc.x >= 400 or loc.y >= 400:
            # This patch should be fully masked
            assert np.all(example_file[loc] == 0)
        elif loc.x <= 200 and loc.y <= 200:
            # This patch should be fully unmasked
            assert np.all(example_file[loc] == example_no_mask[loc])
        else:
            # This patche covers a mix of masked and non-masked areas
            assert np.any(example_file[loc] != 0)

