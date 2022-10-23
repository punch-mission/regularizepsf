from __future__ import annotations

from typing import Callable, TypeAlias, Any, cast, Generic
from numbers import Real
import inspect
from functools import wraps, partial, update_wrapper


import numpy as np
from spectrum import create_window
import dill

from psfpy.exceptions import (ParameterValidationError,
                              InvalidSizeError,
                              EvaluatedModelInconsistentSizeError,
                              UnevaluatedPointError,
                              ParameterMismatchOnEvaluationError,
                              ParameterMismatchOnConstructionError)

Point: TypeAlias = tuple[int, int]


class PSF:
    def __init__(self, function: Callable):
        self._f: Callable = function
        self._signature: inspect.Signature = inspect.signature(function)
        self._parameters: set[str] = set()

        if len(self._signature.parameters) < 2:
            raise ParameterValidationError("x and y must be the first two arguments in your model equation.")

        for i, variable in enumerate(self._signature.parameters):
            if i == 0 and variable != "x":
                raise ParameterValidationError("x must be the first arguments in your model equation.")
            elif i == 1 and variable != "y":
                raise ParameterValidationError("y must be the second arguments in your model equation")
            if i >= 2:
                self._parameters.add(variable)

    def __call__(self, *args, **kwargs) -> Real | np.ndarray:
        return self._f(*args, **kwargs)

    @property
    def parameters(self) -> set[str]:
        return self._parameters


def psf(arg=None) -> PSF:
    if callable(arg):
        return PSF(arg)
    else:
        raise Exception("psf decorator must have no arguments.")


class VariedPSF:
    def __init__(self, vary_function: Callable, base_psf: PSF, check_at_call: bool = True):
        self._vary_function = vary_function
        self._base_psf = base_psf
        self.check_at_call = check_at_call

        self.parameterization_signature = inspect.signature(vary_function)
        if len(self.parameterization_signature.parameters) < 2:
            raise ParameterValidationError(f"Found {len(self.parameterization_signature.parameters)}")

        if len(self.parameterization_signature.parameters) > 2:
            raise ParameterValidationError(
                f"Found function requiring {len(self.parameterization_signature.parameters)} arguments."
                "Expected 2, only `x` and `y`.")

        for i, variable in enumerate(self.parameterization_signature.parameters):
            if i == 0 and variable != "x":
                raise ParameterValidationError("x must be the first argument in your parameterization equation.")
            elif i == 1 and variable != "y":
                raise ParameterValidationError("y must be the second argument in your parameterization equation")

        origin_evaluation: dict[str, Any] = vary_function(0, 0)
        self.parameterization_parameters: set[str] = set(origin_evaluation.keys())
        if self._base_psf.parameters != self.parameterization_parameters:
            msg = (f"The reference function has parameters {self._base_psf.parameters} "
                   f"while the parameterization has {self.parameterization_parameters}. These do not match.")
            raise ParameterMismatchOnConstructionError(msg)

    def __call__(self, *args, **kwargs):
        result = self._vary_function(*args, **kwargs)
        if self.check_at_call:
            if set(result.keys()) != self.parameterization_parameters:
                raise ParameterMismatchOnEvaluationError(f"Evaluated to {set(result.keys())} "
                                                         f"when the parameters were expected as {self.parameterization_parameters}.")
        return result


def _varied_psf(base_psf: PSF):
    if base_psf is None:
        raise Exception("A base_psf must be provided to the varied_psf decorator.")

    def inner(__fn=None, *, check_at_call: bool = True):
        if __fn:
            return VariedPSF(__fn, base_psf, check_at_call=check_at_call)
        else:
            return partial(inner, check_at_call=check_at_call)
    return inner


def varied_psf(base_psf: PSF = None) -> VariedPSF:
    if isinstance(base_psf, PSF):
        return cast(VariedPSF, _varied_psf(base_psf))
    else:
        if callable(base_psf):
            raise Exception("varied_psf decorator must be called with an argument for the base_psf.")
        else:
            raise Exception("varied_psf decorator expects exactly one argument of type PSF.")


class FunctionalCorrector:
    def __init__(self, equation: psf,
                 parameterization: VariedPSF | None,
                 target_model: PSF | None):
        self._base_model: equation = equation
        self.variable = parameterization is not None
        self._parameterization: VariedPSF | None = parameterization
        self._target_model = target_model

    def evaluate(self, x: np.ndarray, y: np.ndarray, size: int) -> ArrayCorrector:
        grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
        evaluations = dict()
        for xx in x:
            for yy in y:
                if self.variable:
                    evaluations[(xx, yy)] = self._base_model(grid_x, grid_y, **self._parameterization(x, y))
                else:
                    evaluations[(xx, yy)] = self._base_model(grid_x, grid_y)

        target_evaluation = self._target_model(grid_x, grid_y)
        return ArrayCorrector(evaluations, target_evaluation)

    def correct_image(self, image: np.ndarray, size: int = None,
                      alpha: float = 0.5, epsilon: float = 0.05, use_gpu: bool = False) -> np.ndarray:
        return self.evaluate(None, None, size).correct_image(image, size=size, alpha=alpha,
                                                             epsilon=epsilon, use_gpu=use_gpu)

    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return dill.load(f)


class ArrayCorrector:
    def __init__(self, evaluations: dict[Point, np.ndarray], target_evaluation: np.ndarray):
        self._evaluation_points: list[Point] = list(evaluations.keys())

        self._size = evaluations[self._evaluation_points[0]].shape[0]
        if self._size <= 0:
            raise InvalidSizeError("Found size of {self._size}. Must be >= 1")

        self._evaluations: dict[Point, np.ndarray] = evaluations
        for (x, y), evaluation in self._evaluations.items():
            if evaluation.shape != (self._size, self._size):
                raise EvaluatedModelInconsistentSizeError(f"Expected evaluated model to have shapes of "
                                                          f"{(self._size, self._size)}. Found {evaluation.shape} "
                                                          f"at {(x, y)}.")

        self._target_evaluation = target_evaluation

    def correct_image(self, image: np.ndarray, size: int = None,
                      alpha: float = 0.5, epsilon: float = 0.05, use_gpu: bool = False) -> np.ndarray:
        # assert len(image.shape) == 2, "img must be a 2-d numpy array."
        # psf_i_shape = next(iter(psf_i.values())).shape
        # assert len(psf_i_shape) == 2, "psf_i entries must be 2-d numpy arrays."
        # assert psf_i_shape[0] == psf_i_shape[1], "PSFs must be square"
        # assert psf_i_shape[0] % 2 == 0, "PSF size must be even in both dimensions"
        # assert all(
        #     v.shape == psf_i_shape for v in psf_i.values()
        # ), "All psf_i entries must be the same shape."
        # assert (
        #     psf_target.shape == psf_i_shape
        # ), "Shapes of psf_i and psf_target do not match."
        # assert all(
        #     img_dim_i >= psf_dim_i for img_dim_i, psf_dim_i in zip(img.shape, psf_i_shape)
        # ), "img must be at least as large as the PSFs in all dimensions"
        if use_gpu:
            try:
                from cupy.fft import fft2, ifft2
            except ImportError as e:
                raise ImportError("cupy is required for GPU acceleration. cupy was not found.")
        else:
            try:
                from numpy.fft import fft2, ifft2
            except ImportError:
                raise ImportError("numpy is found for CPU execution. numpy was not found.")

        psf_size = self._size
        padding_shape = ((psf_size // 2, psf_size // 2), (psf_size // 2, psf_size // 2))
        padded_img = np.pad(image, padding_shape, mode="constant")
        result_img = np.zeros_like(padded_img)

        psf_target_hat = fft2(self._target_evaluation)

        window1d = create_window(psf_size, "cosine")
        apodization_window = np.sqrt(np.outer(window1d, window1d))

        for (x, y), this_psf_i in self._evaluations.items():
            this_psf_i_padded = np.pad(this_psf_i, padding_shape)
            this_psf_i_hat = fft2(this_psf_i_padded)
            this_psf_i_hat_abs = np.abs(this_psf_i_hat)
            this_psf_i_hat_norm = (np.conj(this_psf_i_hat) / this_psf_i_hat_abs) * (
                np.power(this_psf_i_hat_abs, alpha)
                / (np.power(this_psf_i_hat_abs, alpha + 1) + np.power(epsilon, alpha + 1))
            )

            img_i = get_padded_img_section(padded_img, x, y, psf_size)
            img_i_apodized_padded = np.pad(img_i * apodization_window, padding_shape)
            img_i_hat = fft2(img_i_apodized_padded)

            corrected_i = np.real(ifft2(img_i_hat * this_psf_i_hat_norm * psf_target_hat))[
                psf_size:psf_size:
            ]
            corrected_i = corrected_i * apodization_window
            set_padded_img_section(result_img, x, y, psf_size, corrected_i)
        return result_img[
            psf_size // 2: image.shape[0] + psf_size // 2,
            psf_size // 2: image.shape[1] + psf_size // 2,
        ]

    def __getitem__(self, xy: Point) -> np.ndarray:
        if xy in self._evaluation_points:
            return self._evaluations[xy]
        else:
            raise UnevaluatedPointError(f"Model not evaluated at {xy}.")

    def pad(self, new_size: int) -> ArrayCorrector:
        pass


def get_padded_img_section(padded_img, x, y, psf_size) -> np.ndarray:
    x_prime, y_prime = x + psf_size // 2, y + psf_size // 2
    return padded_img[x_prime : x_prime + psf_size, y_prime : y_prime + psf_size]


def set_padded_img_section(padded_img, x, y, psf_size, new_values) -> None:
    assert new_values.shape == (psf_size, psf_size)
    x_prime, y_prime = x + psf_size // 2, y + psf_size // 2
    padded_img[x_prime : x_prime + psf_size, y_prime : y_prime + psf_size] = new_values
