from __future__ import annotations

import abc
import inspect
from functools import partial
from numbers import Real
from typing import Any, Callable, Dict, List, cast

import numpy as np

from regularizepsf.exceptions import (
    PSFParameterValidationError,
    VariedPSFParameterMismatchError,
)


class PointSpreadFunctionABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: Real | np.ndarary, y: Real | np.ndarray) -> Real | np.ndarray:
        """Evaluation of the point spread function

        Parameters
        ----------
        x : real number or `np.ndarray`
            first dimension coordinate to evaluate at
        y : real number or `np.ndarray`
            second dimension coordinate to evaluate at

        Returns
        -------
        real number or `np.ndarray`
            the value of the point spread function at (x,y)
        """

    @property
    @abc.abstractmethod
    def parameters(self) -> List:
        """Varying parameters of the model."""


class SimplePSF(PointSpreadFunctionABC):
    """Model for a simple PSF"""

    def __init__(self, function: Callable) -> None:
        """Creates a PSF object

        Parameters
        ----------
        function
            Python function representing the PSF,
                first two parameters must be x and y and must return an numpy array
        """
        self._f: Callable = function
        self._signature: inspect.Signature = inspect.signature(function)
        self._parameters: set[str] = set()

        if len(self._signature.parameters) < 2:
            msg = "x and y must be the first two arguments in your model equation."
            raise PSFParameterValidationError(msg)

        for i, variable in enumerate(self._signature.parameters):
            if i == 0 and variable != "x":
                msg = "x must be the first arguments in your model equation."
                raise PSFParameterValidationError(msg)
            elif i == 1 and variable != "y":
                msg = "y must be the second arguments in your model equation"
                raise PSFParameterValidationError(msg)
            if i >= 2:
                self._parameters.add(variable)

    def __call__(self, 
                 x: Real | np.ndarray,
                 y: Real | np.ndarray, 
                 **kwargs: Dict[str, Any]) -> Real | np.ndarray:
        return self._f(x, y, **kwargs)

    @property
    def parameters(self) -> set[str]:
        return self._parameters


def simple_psf(arg: Any=None) -> SimplePSF:
    if callable(arg):
        return SimplePSF(arg)
    else:
        raise TypeError("psf decorator must have no arguments.")


class VariedPSF(PointSpreadFunctionABC):
    """Model for a PSF that varies over the field of view"""

    def __init__(self, 
                 vary_function: Callable,
                 base_psf: SimplePSF, 
                 validate_at_call: bool = True) -> None:
        self._vary_function = vary_function
        self._base_psf = base_psf
        self.validate_at_call = validate_at_call

        self.parameterization_signature = inspect.signature(vary_function)
        if len(self.parameterization_signature.parameters) < 2:
            msg = f"Found {len(self.parameterization_signature.parameters)}"
            raise PSFParameterValidationError(msg)

        if len(self.parameterization_signature.parameters) > 2:
            msg = ("Found function requiring"
                   f"{len(self.parameterization_signature.parameters)}"
                   "arguments. Expected 2, only `x` and `y`.")
            raise PSFParameterValidationError(msg)

        for i, variable in enumerate(self.parameterization_signature.parameters):
            if i == 0 and variable != "x":
                msg = "x must be the first argument in your parameterization equation."
                raise PSFParameterValidationError(msg)
            elif i == 1 and variable != "y":
                msg = "y must be the second argument in your parameterization equation"
                raise PSFParameterValidationError(msg)

        # check the parameters at the origin
        origin_evaluation: dict[str, Any] = vary_function(0, 0)
        self._origin_parameters: set[str] = set(origin_evaluation.keys())
        if self._base_psf.parameters != self._origin_parameters:
            msg = (f"The base PSF model has parameters {self._base_psf.parameters} "
                   f"while the varied psf supplies {self._origin_parameters}" 
                   "at the origin. These must match.")
            raise VariedPSFParameterMismatchError(msg)

    def __call__(self, x: Real | np.ndarray, y: Real | np.ndarray) -> Real | np.ndarray:
        variance = self._vary_function(x, y)
        if self.validate_at_call and set(variance.keys()) != self.parameters:
                msg = (f"At (x, y) the varying parameters were {set(variance.keys())}"
                       f" when the parameters were expected as {self.parameters}.")
                raise VariedPSFParameterMismatchError(msg)
        return self._base_psf(x, y, **variance)

    @property
    def parameters(self) -> List:
        return self._base_psf.parameters


def _varied_psf(base_psf: SimplePSF) -> VariedPSF:
    if base_psf is None:
        raise TypeError("A base_psf must be provided to the varied_psf decorator.")

    def inner(__fn: Callable=None, *, check_at_call: bool = True) -> Callable:
        if __fn:
            return VariedPSF(__fn, base_psf, validate_at_call=check_at_call)
        else:
            return partial(inner, check_at_call=check_at_call)

    return inner


def varied_psf(base_psf: SimplePSF = None) -> VariedPSF:
    if isinstance(base_psf, SimplePSF):
        return cast(VariedPSF, _varied_psf(base_psf))
    else:
        if callable(base_psf):
            raise TypeError("varied_psf decorator must be called"
                            "with an argument for the base_psf.")
        else:
            raise TypeError("varied_psf decorator expects exactly"
                            "one argument of type PSF.")
