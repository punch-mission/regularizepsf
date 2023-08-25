import os.path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pytest import fixture

from regularizepsf.corrector import (
    ArrayCorrector,
    FunctionalCorrector,
    calculate_covering,
)
from regularizepsf.exceptions import (
    EvaluatedModelInconsistentSizeError,
    InvalidSizeError,
    UnevaluatedPointError,
)
from regularizepsf.psf import simple_psf, varied_psf


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


def test_create_functional_corrector(tmp_path):
    example = FunctionalCorrector(example_psf, example_psf)
    assert example._psf == example_psf
    assert example.is_variable is False
    assert example._target_model == example_psf

    fname = tmp_path / "test.psf"
    example.save(fname)
    assert os.path.isfile(fname)
    loaded = example.load(fname)
    assert isinstance(loaded, FunctionalCorrector)


def test_evaluate_to_array_form_with_invalid_size_errors():
    @simple_psf
    def base(x, y):
        return np.ones_like(x)

    func_corrector = FunctionalCorrector(base, None)
    with pytest.raises(InvalidSizeError):
        arr_corrector = func_corrector.evaluate_to_array_form(np.arange(10), np.arange(10), 11)


def test_evaluate_to_array_form_with_ones_and_no_target():
    @simple_psf
    def base(x, y):
        return np.ones_like(x)

    func_corrector = FunctionalCorrector(base, None)
    arr_corrector = func_corrector.evaluate_to_array_form(np.arange(10), np.arange(10), 10)
    assert isinstance(arr_corrector, ArrayCorrector)
    assert len(arr_corrector._evaluations) == 100
    assert len(arr_corrector._evaluation_points) == 100
    assert np.all(arr_corrector[0, 0] == 1)


def test_evaluate_to_array_form_with_ones_and_target():
    @simple_psf
    def base(x, y):
        return np.ones_like(x)

    @simple_psf
    def target(x, y):
        return np.ones_like(x).astype(float)


    func_corrector = FunctionalCorrector(base, target)
    arr_corrector = func_corrector.evaluate_to_array_form(np.arange(10), np.arange(10), 10)
    assert isinstance(arr_corrector, ArrayCorrector)
    assert len(arr_corrector._evaluations) == 100
    assert len(arr_corrector._evaluation_points) == 100
    assert np.all(arr_corrector[0, 0] == 1)


def test_functional_corrector_correct_image():
    @simple_psf
    def base(x, y):
        return np.ones_like(x)

    @simple_psf
    def target(x, y):
        return np.ones_like(x).astype(float)

    func_corrector = FunctionalCorrector(base, target)
    raw_image = np.ones((100, 100))
    corrected_image = func_corrector.correct_image(raw_image, 10)
    assert raw_image.shape == corrected_image.shape


def test_array_corrector_without_numpy_arrays():
    evaluations = {(1, 1): 1}
    target = np.ones((100, 100))
    with pytest.raises(TypeError):
        corr = ArrayCorrector(evaluations, target)


def test_array_corrector_correct_image_with_image_smaller_than_psf():
    image = np.ones((10, 10))
    evaluations = {(0, 0): np.ones((100, 100))}
    target = np.ones((100, 100))
    with pytest.raises(InvalidSizeError):
        corr = ArrayCorrector(evaluations, target)
        corr.correct_image(image, 10)


def test_array_corrector_get_nonexistent_point():
    evaluations = {(0, 0): np.ones((100, 100))}
    target = np.ones((100, 100))
    with pytest.raises(UnevaluatedPointError):
        corr = ArrayCorrector(evaluations, target)
        patch = corr[(1, 1)]


def test_save_load_array_corrector(tmp_path):
    evaluations = {(0, 0): np.ones((100, 100))}
    target = np.ones((100, 100))
    example = ArrayCorrector(evaluations, target)
    assert len(example._evaluations) == 1

    fname = tmp_path / "test.psf"
    example.save(fname)
    assert os.path.isfile(fname)
    loaded = example.load(fname)
    assert isinstance(loaded, ArrayCorrector)
    assert np.all(loaded._target_evaluation == np.ones((100, 100)))
    assert np.all(loaded._evaluations[(0,0)] == np.ones((100, 100)))


def test_array_corrector_simulate_observation_with_zero_stars():
    evaluations = {(0, 0): np.ones((100, 100))}
    target = np.ones((100, 100))
    corrector = ArrayCorrector(evaluations, target)

    fake_stars = np.zeros((100, 100))
    fake_observation = corrector.simulate_observation(fake_stars)

    assert isinstance(fake_observation, np.ndarray)
    assert fake_observation.shape == (100, 100)
    assert np.all(fake_observation == 0)


def test_functional_corrector_simulate_observation():
    @simple_psf
    def base(x, y):
        return np.ones_like(x)

    @simple_psf
    def target(x, y):
        return np.ones_like(x).astype(float)

    func_corrector = FunctionalCorrector(base, target)

    fake_stars = np.zeros((100, 100))
    fake_observation = func_corrector.simulate_observation(fake_stars, 100)

    assert isinstance(fake_observation, np.ndarray)
    assert fake_observation.shape == (100, 100)
    assert np.all(fake_observation == 0)
