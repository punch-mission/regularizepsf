import pytest

from psfpy.model import base_equation
from psfpy.exceptions import MissingParameterError


def test_base_equation_valid():
    func = lambda x, y: x + y
    eqn = base_equation(func)
    assert isinstance(eqn, base_equation)
    assert eqn.parameters == []


def test_base_equation_missing_xy():
    func = lambda: 1
    with pytest.raises(MissingParameterError):
        eqn = base_equation(func)


def test_base_equation_misordered_xy():
    func = lambda y, x: x + y
    with pytest.raises(MissingParameterError):
        eqn = base_equation(func)
