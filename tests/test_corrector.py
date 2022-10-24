import pytest
from pytest import fixture
import numpy as np

from psfpy.corrector import calculate_covering, get_padded_img_section, set_padded_img_section


def confirm_full_double_covering(corners, img_shape, patch_size):
    counts = np.zeros(img_shape)
    for i, (x, y) in enumerate(corners):
        counts[np.max([0, x]):np.min([img_shape[0], x + patch_size]),
               np.max([0, y]):np.min([img_shape[1], y + patch_size])] += 1
    assert np.all(counts == 2)


@pytest.mark.parametrize("img_shape, patch_size, expected_count",
                         [((5, 5), 1, 85),
                          ((5, 5), 2, 25),
                          ((15, 15), 3, 85),
                          ((15, 15), 4, 41)])
def test_calculate_covering(img_shape, patch_size, expected_count):
    corners = calculate_covering(img_shape, patch_size)
    assert corners.shape == (expected_count, 2)
    confirm_full_double_covering(corners, img_shape, patch_size)


@fixture
def padded_100by100_image_psf_10_with_pattern():
    img = np.ones((100, 100))
    img[:10, :10] = 2
    padding_shape = ((10, 10), (10, 10))
    img_padded = np.pad(img, padding_shape, mode='constant')
    return img_padded


@pytest.mark.parametrize("coord, value",
                         [((0, 0), 2),
                          ((10, 10), 1),
                          ((-10, -10), 0)])
def test_get_padded_img_section(coord, value, padded_100by100_image_psf_10_with_pattern):
    img_i = get_padded_img_section(padded_100by100_image_psf_10_with_pattern, coord[0], coord[1], 10)
    assert np.all(img_i == np.zeros((10, 10)) + value)


def test_set_padded_img_section(padded_100by100_image_psf_10_with_pattern):
    test_img = np.pad(np.ones((100, 100)), ((10, 10), (10, 10)), mode='constant')
    for coord, value in [((0, 0), 2), ((10, 10), 1), ((-10, -10), 0)]:
        set_padded_img_section(test_img, coord[0], coord[1], 10, np.zeros((10, 10))+value)
    assert np.all(test_img == padded_100by100_image_psf_10_with_pattern)

