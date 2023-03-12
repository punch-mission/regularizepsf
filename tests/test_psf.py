import pytest

from regularizepsf.psf import simple_psf, varied_psf, SimplePSF, VariedPSF
from regularizepsf.exceptions import PSFParameterValidationError, VariedPSFParameterMismatchError


def test_simple_psf_valid():
    """ Confirms that a psf with no extra parameters works"""
    func = lambda x, y: x + y
    eqn = simple_psf(func)
    assert isinstance(eqn, SimplePSF)
    assert eqn.parameters == set()
    assert eqn(1, 2) == 3


def test_simple_psf_two_parameters():
    """ Confirms that a psf with two parameters performs correctly"""
    func = lambda x, y, sigma=3, mu=4: x + y + sigma + mu
    eqn = simple_psf(func)
    assert isinstance(eqn, SimplePSF)
    assert eqn.parameters == {'sigma', 'mu'}
    assert eqn(1, 2) == 10


def test_simple_psf_missing_xy_fails():
    """ Confirms that a psf without x and y arguments fails"""
    with pytest.raises(PSFParameterValidationError):
        simple_psf(lambda: 1)


def test_simple_psf_swap_x_and_y_fails():
    """ Ensures x and y must be in the proper order"""
    with pytest.raises(PSFParameterValidationError):
        simple_psf(lambda y, x: x + y)


def test_simple_psf_missing_y_fails():
    """ Ensures y must be the second argument"""
    with pytest.raises(PSFParameterValidationError):
        simple_psf(lambda x, sigma: x + sigma)


def test_varied_psf_simple_is_valid():
    """ Ensures a simple varied psf performs correctly"""
    base = simple_psf(lambda x, y, sigma=5: x + y + sigma)
    my_psf = varied_psf(base)(lambda x, y: {"sigma": 1})
    assert isinstance(my_psf, VariedPSF)
    assert my_psf.parameters == {'sigma'}
    assert my_psf(0, 0) == 1


def test_varied_psf_too_few_parameters_fails():
    """ Confirms that a varied psf that has too few parameters compared to the base model fails"""
    base = simple_psf(lambda x, y, sigma, mu: x + y)
    with pytest.raises(PSFParameterValidationError):
        varied_psf(base)(lambda: {'sigma': 0.1})


def test_varied_psf_too_many_parameters_fails():
    """ Confirms that a varied psf with too many parameters compared to the base model fails"""
    ref = simple_psf(lambda x, y: x + y)
    with pytest.raises(PSFParameterValidationError):
        varied_psf(ref)(lambda x, y, c: {'sigma': 0.1})


def test_varied_psf_missing_x_fails():
    """ Confirms a varied psf model with a missing x fails"""
    ref = simple_psf(lambda x, y: x + y)
    with pytest.raises(PSFParameterValidationError):
        varied_psf(ref)(lambda c, y: {'sigma': 0.1})


def test_varied_psf_missing_y_fails():
    """ Confirms a varied psf model with a missing y fails"""
    ref = simple_psf(lambda x, y: x + y)
    with pytest.raises(PSFParameterValidationError):
        varied_psf(ref)(lambda x, c: {'sigma': 0.1})


def test_varied_psf_called_without_arguments():
    with pytest.raises(TypeError):
        varied_psf()(lambda x, y: {"sigma": 0.2})


def test_varied_psf_called_with_none_base_psf():
    with pytest.raises(TypeError):
        @varied_psf(None)
        def func(x, y):
            return {"sigma": 0.2}


def test_varied_psf_called_naked():
    with pytest.raises(TypeError):
        @varied_psf
        def func(x, y):
            return {"sigma": 0.1}


def test_varied_psf_parameters_not_match_base_errors():
    @simple_psf
    def base(x, y, m):
        return x + y

    with pytest.raises(VariedPSFParameterMismatchError):
        @varied_psf(base)
        def varied(x, y):
            return {"n": 0, "m": 30}


def test_varied_psf_parameters_match_except_at_call_errors():
    @simple_psf
    def base(x, y, m):
        return x + y

    with pytest.raises(VariedPSFParameterMismatchError):
        @varied_psf(base)
        def varied(x, y):
            if x == 0 and y == 0:
                return {"m": 30}
            else:
                return {"n": 100, "m": 30}
        varied(10, 10)
