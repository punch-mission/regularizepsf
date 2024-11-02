import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from regularizepsf.exceptions import IncorrectShapeError, InvalidCoordinateError
from regularizepsf.util import IndexedCube, calculate_covering


def confirm_full_four_covering(corners, img_shape, patch_size):
    """Confirms that the covering fully covers the image four times, i.e. each point is sample four times."""
    counts = np.zeros(img_shape)
    for i, (x, y) in enumerate(corners):
        counts[np.max([0, x]):np.min([img_shape[0], x + patch_size]),
               np.max([0, y]):np.min([img_shape[1], y + patch_size])] += 1
    assert np.all(counts == 4)


@pytest.mark.parametrize("img_shape, patch_size",
                         [((5, 5), 1),
                          ((5, 5), 2),
                          ((15, 15), 3),
                          ((15, 15), 4),
                          ((100, 100), 11)])
def test_calculate_covering_with_given_sizes(img_shape, patch_size):
    """Calculates a variety of coverings to make sure they're all properly covered."""
    corners = calculate_covering(img_shape, patch_size)
    confirm_full_four_covering(corners, img_shape, patch_size)


@given(img_dim=st.integers(min_value=100, max_value=200), patch_fraction=st.fractions(min_value=0.1, max_value=0.8))
@settings(max_examples=150, deadline=None)
def test_calculate_covering_random_square_images_always_covered(img_dim, patch_fraction):
    """Similar to `test_calculate_covering_with_given_sizes`, but randomly generates square patches for covering."""
    img_shape = (img_dim, img_dim)
    patch_size = np.ceil(img_dim * patch_fraction)
    corners = calculate_covering(img_shape, patch_size)
    confirm_full_four_covering(corners, img_shape, patch_size)


@pytest.mark.parametrize("num_layers, x_shape, y_shape",
                         [(10, 10, 10),
                          (15, 20, 25),
                          (1, 15, 10),
                          (1, 1, 1),
                          (0, 1, 1),
                          (0, 0, 0)])
def test_indexed_cube_general_functionality(num_layers, x_shape, y_shape):
    """Tests that the IndexedCube can be created, has the proper shape and length, and indexes where expected."""
    # construct cube
    data = np.zeros((num_layers, x_shape, y_shape))
    for i in range(num_layers):
        data[i] = i

    coordinates = [(i, i+1) for i in range(num_layers)]

    cube = IndexedCube(coordinates, data)

    # test cube
    assert cube.sample_shape == (x_shape, y_shape)
    assert len(cube) == num_layers
    assert cube.coordinates == coordinates
    for coord in coordinates:
        assert np.all(cube[coord] == coord[0])

        with pytest.raises(InvalidCoordinateError):
            _ = cube[(coord[1], coord[0])]

        with pytest.raises(InvalidCoordinateError):
            cube[(coord[1], coord[0])] = np.zeros((x_shape, y_shape))

        with pytest.raises(IncorrectShapeError):
            cube[coord] = np.zeros((x_shape+1, y_shape+1))

        cube[coord] = np.zeros((x_shape, y_shape))

    assert np.all(cube._values == 0)


def test_compare_indexed_cube_to_other_fails():
    """Cannot compare an IndexedCube to anything other than an IndexedCube."""
    cube = IndexedCube([(0, 0), (0, 1)], np.ones((2, 2, 2)))
    with pytest.raises(TypeError):
        _ = cube == np.zeros((2, 2, 2))

def test_indexed_cube_wrong_coordinate_length_fails():
    """IndexedCube must be self-consistent in coordinate length"""
    with pytest.raises(IncorrectShapeError):
        _ = IndexedCube([(0, 0), (0, 1), (5, 5)], np.ones((2, 2, 2)))

def test_indexed_cube_must_be_3d():
    """IndexedCube must be 3D"""
    with pytest.raises(IncorrectShapeError):
        _ = IndexedCube([(0, 0), (0, 1)], np.ones((2, 2)))
