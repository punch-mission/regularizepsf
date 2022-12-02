import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

from regularizepsf.fitter import CoordinatePatchCollection, CoordinateIdentifier


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
