import os.path

import pytest
from pytest import fixture
import numpy as np
from hypothesis import given, strategies as st, settings

from regularizepsf.psf import simple_psf, varied_psf
from regularizepsf.corrector import calculate_covering, ArrayCorrector, FunctionalCorrector
from regularizepsf.exceptions import InvalidSizeError, EvaluatedModelInconsistentSizeError

def confirm_full_four_covering(corners, img_shape, patch_size):
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
    corners = calculate_covering(img_shape, patch_size)
    confirm_full_four_covering(corners, img_shape, patch_size)


@given(img_dim=st.integers(min_value=100, max_value=200), patch_fraction=st.fractions(min_value=0.1, max_value=0.8))
@settings(max_examples=150, deadline=None)
def test_calculate_covering_random_square_images_always_covered(img_dim, patch_fraction):
    img_shape = (img_dim, img_dim)
    patch_size = np.ceil(img_dim * patch_fraction)
    corners = calculate_covering(img_shape, patch_size)
    confirm_full_four_covering(corners, img_shape, patch_size)


@fixture
def padded_100by100_image_psf_10_with_pattern():
    img = np.ones((100, 100))
    img[:10, :10] = 2
    padding_shape = ((20, 20), (20, 20))
    img_padded = np.pad(img, padding_shape, mode='constant')
    return img_padded

#
# @pytest.mark.parametrize("coord, value",
#                          [((0, 0), 2),
#                           ((10, 10), 1),
#                           ((-10, -10), 0)])
# def test_get_padded_img_section(coord, value, padded_100by100_image_psf_10_with_pattern):
#     img_i = get_padded_img_section(padded_100by100_image_psf_10_with_pattern, coord[0], coord[1], 10)
#     assert np.all(img_i == np.zeros((10, 10)) + value)
#
#
# def test_set_padded_img_section(padded_100by100_image_psf_10_with_pattern):
#     test_img = np.pad(np.ones((100, 100)), ((20, 20), (20, 20)), mode='constant')
#     for coord, value in [((0, 0), 2), ((10, 10), 1), ((-10, -10), 0)]:
#         set_padded_img_section(test_img, coord[0], coord[1], 10, np.zeros((10, 10))+value)
#     assert np.all(test_img == padded_100by100_image_psf_10_with_pattern)


def test_create_array_corrector():
    example = ArrayCorrector({(0, 0): np.zeros((10, 10))},
                             np.zeros((10, 10)))
    assert isinstance(example, ArrayCorrector)
    assert example._evaluation_points == [(0, 0)]


def test_nonimage_array_corrector_errors():
    with pytest.raises(InvalidSizeError):
        example = ArrayCorrector({(0, 0): np.zeros(10)}, np.zeros(10))


def test_noneven_array_corrector_errors():
    with pytest.raises(InvalidSizeError):
        example = ArrayCorrector({(0, 0): np.zeros((11, 11))}, np.zeros((11, 11)))


def test_array_corrector_with_different_size_evaluations_errors():
    with pytest.raises(EvaluatedModelInconsistentSizeError):
        example = ArrayCorrector({(0, 0): np.zeros((10, 10)), (1, 1): np.zeros((20, 20))},
                                 np.zeros((20, 20)))


def test_array_corrector_with_different_size_than_target_errors():
    with pytest.raises(EvaluatedModelInconsistentSizeError):
        example = ArrayCorrector({(0, 0): np.zeros((10, 10)), (1, 1): np.zeros((10, 10))},
                                 np.zeros((20, 20)))


@simple_psf
def example_psf(x, y):
    return 0


def test_create_functional_corrector():
    example = FunctionalCorrector(example_psf, example_psf)
    assert example._psf == example_psf
    assert example.is_variable is False
    assert example._target_model == example_psf

    example.save("test.psf")
    assert os.path.isfile("test.psf")
    loaded = example.load("test.psf")
    assert isinstance(loaded, FunctionalCorrector)
    os.remove("test.psf")
