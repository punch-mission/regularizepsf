import pytest

from psfpy.model import base_equation, base_parameterization
from psfpy.exceptions import ParameterValidationError


def test_base_equation_valid():
    func = lambda x, y: x + y
    eqn = base_equation(func)
    assert isinstance(eqn, base_equation)
    assert eqn.parameters == set()
    assert eqn(1, 2) == 3


def test_base_equation_many_parameters():
    func = lambda x, y, sigma, mu: x + y + sigma + mu
    eqn = base_equation(func)
    assert isinstance(eqn, base_equation)
    assert eqn.parameters == {'sigma', 'mu'}
    assert eqn(1, 2, 3, 4) == 10


def test_base_equation_missing_xy():
    func = lambda: 1
    with pytest.raises(ParameterValidationError):
        eqn = base_equation(func)


def test_base_equation_misordered_xy():
    func = lambda y, x: x + y
    with pytest.raises(ParameterValidationError):
        eqn = base_equation(func)


def test_base_equation_missing_y():
    func = lambda x, sigma: x + sigma
    with pytest.raises(ParameterValidationError):
        eqn = base_equation(func)


def test_base_parameterization_valid():
    ref = base_equation(lambda x, y, sigma: x + y)
    func = lambda x, y: {"sigma": 0.1}
    parameterization = base_parameterization(func, reference_function=ref)
    assert isinstance(parameterization, base_parameterization)
    assert parameterization._parameters == {'sigma'}
    assert parameterization(0, 0) == {"sigma": 0.1}


def test_base_parameterization_too_few_parameters_failure():
    ref = base_equation(lambda x, y: x + y)
    func = lambda: {'sigma': 0.1}
    with pytest.raises(ParameterValidationError):
        parameterization = base_parameterization(func, reference_function=ref)


def test_base_parameterization_too_many_parameters_failure():
    ref = base_equation(lambda x, y: x + y)
    func = lambda x, y, c: {'sigma': 0.1}
    with pytest.raises(ParameterValidationError):
        parameterization = base_parameterization(func, reference_function=ref)


def test_base_parameterization_missing_x_failure():
    ref = base_equation(lambda x, y: x + y)
    func = lambda c, y: {'sigma': 0.1}
    with pytest.raises(ParameterValidationError):
        parameterization = base_parameterization(func, reference_function=ref)


def test_base_parameterization_missing_y_failure():
    ref = base_equation(lambda x, y: x + y)
    func = lambda x, c: {'sigma': 0.1}
    with pytest.raises(ParameterValidationError):
        parameterization = base_parameterization(func, reference_function=ref)
