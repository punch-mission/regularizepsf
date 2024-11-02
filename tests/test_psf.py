import numpy as np
import pytest
import scipy

from regularizepsf.exceptions import IncorrectShapeError, InvalidCoordinateError, InvalidFunctionError
from regularizepsf.psf import (
    ArrayPSF,
    SimpleFunctionalPSF,
    VariedFunctionalPSF,
    simple_functional_psf,
    varied_functional_psf,
)
from regularizepsf.util import IndexedCube
from tests.helper import make_gaussian


@pytest.mark.parametrize("extension", ["fits", "h5"])
def test_arraypsf_saves_and_loads(tmp_path, extension):
    """Can save and reload an ArrayPSF"""
    coordinates = [(0, 0), (1, 1), (2, 2)]
    gauss = make_gaussian(128, fwhm=3)
    values = np.stack([gauss for _ in coordinates])

    source = ArrayPSF(IndexedCube(coordinates, values))

    path = tmp_path / f"psf.{extension}"

    source.save(path)
    reloaded = ArrayPSF.load(path)

    assert source == reloaded

def test_arraypsf_compare_to_array_fails():
    """Can only compare ArrayPSF to an ArrayPSF."""
    coordinates = [(0, 0), (1, 1), (2, 2)]
    gauss = make_gaussian(128, fwhm=3)
    values = np.stack([gauss for _ in coordinates])

    source = ArrayPSF(IndexedCube(coordinates, values))

    with pytest.raises(TypeError):
        _ = source == np.zeros((50, 50))


def test_arraypsf_with_mismatched_coordinates_fails():
    """ArrayPSF values and shapes must have same coordinates."""
    coordinates = [(0, 0), (1, 1), (2, 2)]
    gauss = make_gaussian(128, fwhm=3)
    values = np.stack([gauss for _ in coordinates])

    with pytest.raises(InvalidCoordinateError):
        _ = ArrayPSF(IndexedCube(coordinates, values), IndexedCube([(0, 0), (1, 1), (3, 3)], values))


def test_arraypsf_with_different_len_fails():
    """ArrayPSFs must have the same len for values and ffts."""
    coordinates = [(0, 0), (1, 1), (2, 2)]
    gauss = make_gaussian(128, fwhm=3)
    values = np.stack([gauss for _ in coordinates])

    fft_coordinates = [(0, 0), (1, 1), (2, 2), (3, 3)]
    gauss = make_gaussian(128, fwhm=3)
    fft_values = np.stack([gauss for _ in fft_coordinates])

    with pytest.raises(IncorrectShapeError):
        _ = ArrayPSF(IndexedCube(coordinates, values), IndexedCube(fft_coordinates, fft_values))


def test_arraypsf_with_different_sample_shapes_fails():
    """ArrayPSFs must have the same shape for values and ffts."""
    coordinates = [(0, 0), (1, 1), (2, 2)]
    gauss = make_gaussian(128, fwhm=3)
    values = np.stack([gauss for _ in coordinates])

    gauss = make_gaussian(64, fwhm=3)
    fft_values = np.stack([gauss for _ in coordinates])

    with pytest.raises(IncorrectShapeError):
        _ = ArrayPSF(IndexedCube(coordinates, values), IndexedCube(coordinates, fft_values))


def test_arraypsf_get_evaluations():
    """Check that evaluations and fft evaluations can be retrieved."""
    coordinates = [(0, 0), (1, 1), (2, 2)]
    gauss = make_gaussian(128, fwhm=3)
    values = np.stack([gauss for _ in coordinates])
    psf = ArrayPSF(IndexedCube(coordinates, values))
    assert np.all(psf[(0, 0)] == gauss)
    assert np.all(psf.fft_at((0, 0)) == scipy.fft.fft2(gauss))



def test_simple_psf_valid():
    """ Confirms that a psf with no extra parameters works"""
    func = lambda row, col: row + col
    eqn = simple_functional_psf(func)
    assert isinstance(eqn, SimpleFunctionalPSF)
    assert eqn.parameters == set()
    assert eqn(1, 2) == 3


def test_simple_psf_two_parameters():
    """ Confirms that a psf with two parameters performs correctly"""
    func = lambda row, col, sigma=3, mu=4: row + col + sigma + mu
    eqn = simple_functional_psf(func)
    assert isinstance(eqn, SimpleFunctionalPSF)
    assert eqn.parameters == {'sigma', 'mu'}
    assert eqn(1, 2) == 10


def test_simple_psf_missing_xy_fails():
    """ Confirms that a psf without x and y arguments fails"""
    with pytest.raises(InvalidFunctionError):
        simple_functional_psf(lambda: 1)


def test_simple_psf_swap_x_and_y_fails():
    """ Ensures x and y must be in the proper order"""
    with pytest.raises(InvalidFunctionError):
        simple_functional_psf(lambda y, x: x + y)


def test_simple_psf_missing_y_fails():
    """ Ensures y must be the second argument"""
    with pytest.raises(InvalidFunctionError):
        simple_functional_psf(lambda x, sigma: x + sigma)


def test_varied_psf_simple_is_valid():
    """ Ensures a simple varied psf performs correctly"""
    base = simple_functional_psf(lambda row, col, sigma=5: row + col + sigma)
    my_psf = varied_functional_psf(base)(lambda row, col: {"sigma": 1})
    assert isinstance(my_psf, VariedFunctionalPSF)
    assert my_psf.parameters == {'sigma'}
    assert my_psf(0, 0) == 1


def test_varied_psf_too_few_parameters_fails():
    """ Confirms that a varied psf that has too few parameters compared to the base model fails"""
    base = simple_functional_psf(lambda row, col, sigma, mu: row + col)
    with pytest.raises(InvalidFunctionError):
        varied_functional_psf(base)(lambda: {'sigma': 0.1})


def test_varied_psf_too_many_parameters_fails():
    """ Confirms that a varied psf with too many parameters compared to the base model fails"""
    ref = simple_functional_psf(lambda row, col: row + col)
    with pytest.raises(InvalidFunctionError):
        varied_functional_psf(ref)(lambda row, col, c: {'sigma': 0.1})


def test_varied_psf_missing_x_fails():
    """ Confirms a varied psf model with a missing x fails"""
    ref = simple_functional_psf(lambda row, col: row + col)
    with pytest.raises(InvalidFunctionError):
        varied_functional_psf(ref)(lambda c, col: {'sigma': 0.1})


def test_varied_psf_missing_y_fails():
    """ Confirms a varied psf model with a missing y fails"""
    ref = simple_functional_psf(lambda row, col: row + col)
    with pytest.raises(InvalidFunctionError):
        varied_functional_psf(ref)(lambda row, c: {'sigma': 0.1})


def test_varied_psf_called_without_arguments():
    with pytest.raises(TypeError):
        varied_functional_psf()(lambda row, col: {"sigma": 0.2})


def test_varied_psf_called_with_none_base_psf():
    with pytest.raises(TypeError):
        @varied_functional_psf(None)
        def func(row, col):
            return {"sigma": 0.2}


def test_varied_psf_called_naked():
    with pytest.raises(TypeError):
        @varied_functional_psf
        def func(row, col):
            return {"sigma": 0.1}


def test_varied_psf_parameters_not_match_base_errors():
    @simple_functional_psf
    def base(row, col, m):
        return row + col

    with pytest.raises(InvalidFunctionError):
        @varied_functional_psf(base)
        def varied(row, col):
            return {"n": 0, "m": 30}


def test_varied_psf_parameters_match_except_at_call_errors():
    @simple_functional_psf
    def base(row, col, m):
        return row + col

    with pytest.raises(InvalidFunctionError):
        @varied_functional_psf(base)
        def varied(row, col):
            if row == 0 and col == 0:
                return {"m": 30}
            else:
                return {"n": 100, "m": 30}
        _ = varied(10, 10)

def test_evaluate_simplefunctionalpsf_to_arraypsf():
    """Can evaluate a simple functional psf to an array psf."""
    def f(row, col, a=10):
        return row + col + a

    functionalpsf = simple_functional_psf(f)
    arraypsf = functionalpsf.as_array_psf([(0, 0), (1, 0)], 3)

    assert len(arraypsf) == 2
    assert arraypsf.sample_shape == (3, 3)
    assert arraypsf.coordinates == [(0, 0), (1, 0)]

    rr, cc = np.meshgrid(np.arange(3), np.arange(3))
    assert np.allclose(arraypsf[(0, 0)], rr + cc + 10)

def test_evaluate_variedfunctionalpsf_to_arraypsf():
    """Can evaluate a varied psf to an array psf."""
    base = simple_functional_psf(lambda row, col, sigma=5: row + col + sigma)
    my_psf = varied_functional_psf(base)(lambda row, col: {"sigma": row*col})

    arraypsf = my_psf.as_array_psf([(0, 0), (3, 4)], 3)

    assert len(arraypsf) == 2
    assert arraypsf.sample_shape == (3, 3)
    assert arraypsf.coordinates == [(0, 0), (3, 4)]

    rr, cc = np.meshgrid(np.arange(3), np.arange(3))
    assert np.allclose(arraypsf[(3, 4)], rr + cc + 3*4)
