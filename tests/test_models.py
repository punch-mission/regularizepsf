from regularizepsf.models import constrained_gaussian, elliptical_gaussian
from regularizepsf.psf import SimplePSF


def test_constrained_gaussian():
    assert isinstance(constrained_gaussian, SimplePSF)
    assert constrained_gaussian(0, 0) == 1


def test_elliptical_gaussain():
    assert isinstance(elliptical_gaussian, SimplePSF)
    assert elliptical_gaussian(0, 0) == 1