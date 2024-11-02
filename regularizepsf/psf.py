"""Representations of point spread functions."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast
from functools import partial

import h5py
import matplotlib as mpl
import numpy as np
import scipy.fft
from astropy.io import fits

from regularizepsf.exceptions import IncorrectShapeError, InvalidCoordinateError, InvalidFunctionError
from regularizepsf.util import IndexedCube
from regularizepsf.visualize import KERNEL_IMSHOW_ARGS_DEFAULT, PSF_IMSHOW_ARGS_DEFAULT, visualize_grid

if TYPE_CHECKING:
    import pathlib
    from numbers import Real
    from collections.abc import Callable


class SimpleFunctionalPSF:
    """Model for a simple PSF."""

    def __init__(self, function: Callable) -> None:
        """Create a PSF object.

        Parameters
        ----------
        function
            Python function representing the PSF,
                first two parameters must be x and y and must return an numpy array

        """
        self._f: Callable = function
        self._signature: inspect.Signature = inspect.signature(function)
        self._parameters: set[str] = set()

        if len(self._signature.parameters) < 2:  # noqa: PLR2004
            msg = "row and col must be the first two arguments in your model equation."
            raise InvalidFunctionError(msg)

        for i, variable in enumerate(self._signature.parameters):
            if i == 0 and variable != "row":
                msg = "row must be the first arguments in your model equation."
                raise InvalidFunctionError(msg)
            if i == 1 and variable != "col":
                msg = "col must be the second arguments in your model equation"
                raise InvalidFunctionError(msg)
            if i >= 2:  # noqa: PLR2004
                self._parameters.add(variable)

    def __call__(self, row: Real | np.ndarray, col: Real | np.ndarray, **kwargs: dict[str, Any]) -> Real | np.ndarray:
        """Get the PSF value at (row, col)."""
        return self._f(row, col, **kwargs)

    @property
    def parameters(self) -> set[str]:
        """Get the parameters of this PSF."""
        return self._parameters

    def as_array_psf(self, coordinates: list[tuple[int, int]], size: int, **kwargs) -> ArrayPSF:  # noqa: ANN003
        """Convert FunctionalPSF to an ArrayPSF."""
        rr, cc = np.meshgrid(np.arange(size), np.arange(size))
        evaluation = self(rr, cc, **kwargs)
        values = [evaluation for _ in coordinates]
        return ArrayPSF(IndexedCube(coordinates, np.stack(values)))

    @property
    def f(self) -> Callable:
        """Retrieve the PSF functional form for calling."""
        return self._f


def simple_functional_psf(arg: Any = None) -> SimpleFunctionalPSF:
    """Decorate a SimpleFunctionalPSF."""
    if callable(arg):
        return SimpleFunctionalPSF(arg)
    msg = "psf decorator must have no arguments."
    raise TypeError(msg)


class VariedFunctionalPSF:
    """Model for a PSF that varies over the field of view."""

    def __init__(self, vary_function: Callable, base_psf: SimpleFunctionalPSF, validate_at_call: bool = True) -> None:
        """Create a VariedFunctionalPSF object.

        Parameters
        ----------
        vary_function : Callable
            function used to vary the parameters of the base_psf
        base_psf : Callable
            base form of the PSF
        validate_at_call : bool
            whether to check if parameters are valid at each call, turning off may be faster but is risky

        """
        self._vary_function = vary_function
        self._base_psf = base_psf
        self.validate_at_call = validate_at_call

        self.parameterization_signature = inspect.signature(vary_function)
        if len(self.parameterization_signature.parameters) < 2:  # noqa: PLR2004
            msg = f"Found {len(self.parameterization_signature.parameters)}"
            raise InvalidFunctionError(msg)

        if len(self.parameterization_signature.parameters) > 2:  # noqa: PLR2004
            msg = (
                "Found function requiring"
                f"{len(self.parameterization_signature.parameters)}"
                "arguments. Expected 2, only `row` and `col`."
            )
            raise InvalidFunctionError(msg)

        for i, variable in enumerate(self.parameterization_signature.parameters):
            if i == 0 and variable != "row":
                msg = "row must be the first argument in your parameterization equation."
                raise InvalidFunctionError(msg)
            if i == 1 and variable != "col":
                msg = "col must be the second argument in your parameterization equation"
                raise InvalidFunctionError(msg)

        # check the parameters at the origin
        origin_evaluation: dict[str, Any] = vary_function(0, 0)
        self._origin_parameters: set[str] = set(origin_evaluation.keys())
        if self._base_psf.parameters != self._origin_parameters:
            msg = (
                f"The base PSF model has parameters {self._base_psf.parameters} "
                f"while the varied psf supplies {self._origin_parameters}"
                "at the origin. These must match."
            )
            raise InvalidFunctionError(msg)

    def __call__(self, row: Real | np.ndarray, col: Real | np.ndarray) -> Real | np.ndarray:
        """Get the PSF value at (row, col)."""
        variance = self._vary_function(row, col)
        if self.validate_at_call and set(variance.keys()) != self.parameters:
            msg = (
                f"At (row, col) the varying parameters were {set(variance.keys())}"
                f" when the parameters were expected as {self.parameters}."
            )
            raise InvalidFunctionError(msg)
        return self._base_psf(row, col, **variance)

    @property
    def parameters(self) -> set[str]:
        """Get the parameters of this PSF."""
        return self._base_psf.parameters

    def simplify(self, row: int, col: int) -> SimpleFunctionalPSF:
        """Simplify this VariedFunctionalPSF to a SimpleFunctionalPSF by evaluating at (row, col)."""
        variance = self._vary_function(row, col)
        return simple_functional_psf(partial(self._base_psf.f, **variance))

    def as_array_psf(self, coordinates: list[tuple[int, int]], size: int, **kwargs) -> ArrayPSF:  # noqa: ANN003
        """Convert FunctionalPSF to an ArrayPSF."""
        values = []
        rr, cc = np.meshgrid(np.arange(size), np.arange(size))
        for row, col in coordinates:
            values.append(self.simplify(row, col)(rr, cc, **kwargs))
        return ArrayPSF(IndexedCube(coordinates, np.stack(values)))


def _varied_functional_psf(base_psf: SimpleFunctionalPSF) -> VariedFunctionalPSF:
    if base_psf is None:
        msg = "A base_psf must be provided to the varied_psf decorator."
        raise TypeError(msg)

    def inner(__fn: Callable = None, *, check_at_call: bool = True) -> Callable:  # noqa: RUF013
        if __fn:
            return VariedFunctionalPSF(__fn, base_psf, validate_at_call=check_at_call)
        return partial(inner, check_at_call=check_at_call)

    return inner


def varied_functional_psf(base_psf: SimpleFunctionalPSF = None) -> VariedFunctionalPSF:
    """Decorate to create a VariedFunctionalPSF."""
    if isinstance(base_psf, SimpleFunctionalPSF):
        return cast(VariedFunctionalPSF, _varied_functional_psf(base_psf))
    if callable(base_psf):
        msg = "varied_psf decorator must be calledwith an argument for the base_psf."
        raise TypeError(msg)
    msg = "varied_psf decorator expects exactlyone argument of type PSF."
    raise TypeError(msg)


class ArrayPSF:
    """A PSF represented as a set of arrays."""

    def __init__(
        self, values_cube: IndexedCube, fft_cube: IndexedCube | None = None, workers: int | None = None,
    ) -> None:
        """Initialize an ArrayPSF model.

        Parameters
        ----------
        values_cube : IndexedCube
            PSF model where keys are upper left coordinates of array patches in the image
        fft_cube : IndexedCube
            fft of the model
        workers: int | None
            Maximum number of workers to use for parallel computation of FFT.
            If negative, the value wraps around from os.cpu_count(). See scipy.fft.fft for more details.
            Only used if fft_cube is None.

        """
        self._values_cube = values_cube
        self._fft_cube = fft_cube
        self._workers = workers

        if self._fft_cube is None:
            self._fft_cube = IndexedCube(
                values_cube.coordinates, scipy.fft.fft2(values_cube.values, workers=self._workers),
            )

        if self._fft_cube.sample_shape != self._values_cube.sample_shape:
            msg = (
                f"Values cube and FFT cube have different sample shapes: "
                f"{self._values_cube.sample_shape} != {self._fft_cube.sample_shape}."
            )
            raise IncorrectShapeError(msg)

        if len(self._fft_cube) != len(self._values_cube):
            msg = (
                f"Values cube and FFT cube have different sample counts: "
                f"{len(self._values_cube)} != {len(self._fft_cube)}."
            )
            raise IncorrectShapeError(msg)

        if np.any(np.array(self._values_cube.coordinates) != np.array(self._fft_cube.coordinates)):
            msg = "Values cube and FFT cube have different coordinates"
            raise InvalidCoordinateError(msg)

    @property
    def coordinates(self) -> list[tuple[int, int]]:
        """Get the keys of the PSF model, i.e., where it is evaluated as an array."""
        return self._values_cube.coordinates

    @property
    def values(self) -> np.ndarray:
        """Get the model values."""
        return self._values_cube.values

    @property
    def fft_evaluations(self) -> np.ndarray:
        """Get the model values."""
        return self._fft_cube.values

    def __getitem__(self, coord: tuple[int, int]) -> np.ndarray:
        """Evaluate the PSF model at specific coordinates."""
        return self._values_cube[coord]

    def fft_at(self, coord: tuple[int, int]) -> np.ndarray:
        """Retrieve the FFT evaluation at a coordinate."""
        return self._fft_cube[coord]

    def save(self, path: pathlib.Path) -> None:
        """Save the PSF model to a file. Supports h5 and FITS.

        Parameters
        ----------
        path : pathlib.Path
            where to save the PSF model

        Returns
        -------
        None

        """
        if path.suffix == ".h5":
            with h5py.File(path, "w") as f:
                f.create_dataset("coordinates", data=self.coordinates)
                f.create_dataset("values", data=self.values)
                f.create_dataset("fft_evaluations", data=self.fft_evaluations)
        elif path.suffix == ".fits":
            fits.HDUList([fits.PrimaryHDU(),
                          fits.CompImageHDU(np.array(self.coordinates), name="coordinates"),
                          fits.CompImageHDU(self.values, name="values"),
                          fits.CompImageHDU(self.fft_evaluations.real, name="fft_real", quantize_level=32),
                          fits.CompImageHDU(self.fft_evaluations.imag, name="fft_imag", quantize_level=32),
                          ]).writeto(path)
        else:
            raise NotImplementedError(f"Unsupported file type {path.suffix}. Change to .h5 or .fits.")

    @classmethod
    def load(cls, path: pathlib.Path) -> ArrayPSF:
        """Load the PSF model from a file. Supports h5 and FITS.

        Parameters
        ----------
        path : pathlib.Path
            where to load the PSF model from

        Returns
        -------
        ArrayPSF
            loaded model

        """
        if path.suffix == ".h5":
            with h5py.File(path, "r") as f:
                coordinates = [tuple(c) for c in f["coordinates"][:]]
                values = f["values"][:]
                fft_evaluations = f["fft_evaluations"][:]
            values_cube = IndexedCube(coordinates, values)
            fft_cube = IndexedCube(coordinates, fft_evaluations)
        elif path.suffix == ".fits":
            with fits.open(path) as hdul:
                coordinates_index = hdul.index_of("coordinates")
                coordinates = [tuple(c) for c in hdul[coordinates_index].data]

                values_index = hdul.index_of("values")
                values = hdul[values_index].data
                values_cube = IndexedCube(coordinates, values)

                fft_real_index = hdul.index_of("fft_real")
                fft_real = hdul[fft_real_index].data
                fft_imag_index = hdul.index_of("fft_imag")
                fft_imag = hdul[fft_imag_index].data
                fft_cube = IndexedCube(coordinates, fft_real + fft_imag*1j)
        else:
            raise NotImplementedError(f"Unsupported file type {path.suffix}. Change to .h5 or .fits.")
        return cls(values_cube, fft_cube)

    def visualize_psfs(self,
                  fig: mpl.figure.Figure | None = None,
                  fig_scale: int = 1,
                  all_patches: bool = False, imshow_args: dict | None = None) -> None:  # noqa: ANN002, ANN003
        """Visualize the PSF model."""
        imshow_args = PSF_IMSHOW_ARGS_DEFAULT if imshow_args is None else imshow_args
        visualize_grid(self._values_cube, fig=fig, fig_scale=fig_scale, all_patches=all_patches,
                       colorbar_label="Normalized brightness",
                       imshow_args=imshow_args)

    def visualize_ffts(self,
                  fig: mpl.figure.Figure | None = None,
                  fig_scale: int = 1,
                  all_patches: bool = False, imshow_args: dict | None = None) -> None:  # noqa: ANN002, ANN003
        """Visualize the fft of the PSF."""
        imshow_args = KERNEL_IMSHOW_ARGS_DEFAULT if imshow_args is None else imshow_args

        arr = np.abs(np.fft.fftshift(np.fft.ifft2(self._fft_cube.values)))
        extent = np.max(np.abs(arr))
        if 'vmin' not in imshow_args:
            imshow_args['vmin'] = -extent
        if 'vmax' not in imshow_args:
            imshow_args['vmax'] = extent

        return visualize_grid(
            IndexedCube(self._fft_cube.coordinates, arr),
            all_patches=all_patches, fig=fig,
            fig_scale=fig_scale, colorbar_label="Transfer kernel amplitude",
            imshow_args=imshow_args)

    def __eq__(self, other: ArrayPSF) -> bool:
        """Check equality between two ArrayPSFs."""
        if not isinstance(other, ArrayPSF):
            msg = "Can only compare ArrayPSF to other ArrayPSF."
            raise TypeError(msg)
        return self._values_cube == other._values_cube and self._fft_cube == other._fft_cube

    @property
    def sample_shape(self) -> tuple[int, int]:
        """Get the sample shape for this PSF model."""
        return self._values_cube.sample_shape

    def __len__(self) -> int:
        """Get the number of coordinates evaluated in this model."""
        return len(self._values_cube)
