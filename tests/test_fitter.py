import os.path
import pathlib

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
    assert cpc._size is None


def test_coordinate_patch_collection_extraction_one_coordinate(incrementing_5_image_set_100x100):
    cpc = CoordinatePatchCollection.extract(incrementing_5_image_set_100x100, [CoordinateIdentifier(0, 0, 0)], 10)
    assert len(cpc) == 1
    assert cpc._size == 10
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
    assert cpc._size == 10
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
    collection = CoordinatePatchCollection({CoordinateIdentifier(0, 0, 0): np.zeros((10, 10)),
                                            CoordinateIdentifier(0, 0, 0): np.ones((10, 10))*2})
    averaged_collection = collection.average(np.array([[0, 0]]), 10, 10, mode='median')
    assert averaged_collection[CoordinateIdentifier(None, 0, 0)][1, 1] == 1


def test_calculate_pad_shape():
    collection = CoordinatePatchCollection({CoordinateIdentifier(0, 0, 0): np.zeros((10, 10))})
    assert collection._size == 10
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
        CoordinatePatchCollection._validate_average_mode("nonexistent_method")


def test_find_stars_and_average():
    img_path = str(TEST_DIR / "data/DASH.fits")
    example = CoordinatePatchCollection.find_stars_and_average([img_path], 32, 100)
    assert isinstance(example, CoordinatePatchCollection)
    for loc, patch in example._patches.items():
        assert patch.shape == (100, 100)


def test_find_stars_and_average_powers_of_2():
    img_path = str(TEST_DIR / "data/DASH.fits")
    example = CoordinatePatchCollection.find_stars_and_average([img_path], 32, 128)
    assert isinstance(example, CoordinatePatchCollection)
    for loc, patch in example._patches.items():
        assert patch.shape == (128, 128)


def test_find_stars_and_average_powers_of_2_mean():
    img_path = str(TEST_DIR / "data/DASH.fits")
    example = CoordinatePatchCollection.find_stars_and_average([img_path], 32, 128, average_mode='mean')
    assert isinstance(example, CoordinatePatchCollection)
    for loc, patch in example._patches.items():
        assert patch.shape == (128, 128)

