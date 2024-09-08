import numpy as np
import pytest

from regularizepsf.exceptions import InvalidCoordinateError
from regularizepsf.psf import ArrayPSF
from regularizepsf.transform import ArrayPSFTransform
from regularizepsf.util import IndexedCube, calculate_covering
from tests.helper import make_gaussian


def test_transform_apply():
    """Test that applying an identity transform does not change the values."""
    size = 256
    gauss = make_gaussian(size, fwhm=3)
    dtype = np.float32

    covering = [tuple(t) for t in calculate_covering((2048, 2048), size)]
    values = np.stack([np.zeros((size, size), dtype=dtype) for _ in covering])
    values[:] = gauss / np.sum(gauss)

    cube = IndexedCube(covering, values)
    source = ArrayPSF(cube, workers=None)

    t = ArrayPSFTransform.construct(source, source, 3.0, 0.1)

    image = np.zeros((2048, 2048), dtype=dtype)
    image[500:1000, 200:400] = 5

    out = t.apply(image)

    assert np.allclose(image, out, atol=1E-3)


def test_transform_with_mismatch_coordinates_errors():
    source_coordinates = [(0, 0), (1, 1), (2, 2)]
    target_coordinates = [(0, 0), (1, 1), (0.5, 0.5)]
    source = ArrayPSF(IndexedCube(source_coordinates, np.zeros((len(source_coordinates), 128, 128))))
    target = ArrayPSF(IndexedCube(target_coordinates, np.zeros((len(target_coordinates), 128, 128))))
    with pytest.raises(InvalidCoordinateError):
        ArrayPSFTransform.construct(source, target, 3.0, 0.1)


def test_transform_save_load(tmp_path):
    path = tmp_path / "transform.h5"
    coordinates = [(0, 0), (1, 1), (2, 2)]
    gauss = make_gaussian(128, fwhm=3)
    values = np.stack([gauss for _ in coordinates])

    source = ArrayPSF(IndexedCube(coordinates, values))
    transform = ArrayPSFTransform.construct(source, source, 3.0, 0.1)

    transform.save(path)
    reloaded = ArrayPSFTransform.load(path)

    assert transform == reloaded


def test_transform_compare_to_array_fails():
    coordinates = [(0, 0), (1, 1), (2, 2)]
    gauss = make_gaussian(128, fwhm=3)
    values = np.stack([gauss for _ in coordinates])

    source = ArrayPSF(IndexedCube(coordinates, values))
    transform = ArrayPSFTransform.construct(source, source, 3.0, 0.1)

    with pytest.raises(TypeError):
        _ = transform == np.zeros((50, 50))
