from __future__ import annotations

from typing import Callable, TypeAlias, Any
from numbers import Real
import inspect

import numpy as np

from psfpy.exceptions import (ParameterValidationError,
                              InvalidSizeError,
                              EvaluatedModelInconsistentSizeError,
                              UnevaluatedPointError)

Point: TypeAlias = tuple[int, int]


class base_equation:
    def __init__(self, function: Callable):
        self.function: Callable = function
        self.signature: inspect.Signature = inspect.signature(function)
        self.parameters: list[str] = []

        if len(self.signature.parameters) < 2:
            raise ParameterValidationError("x and y must be the first two arguments in your model equation.")

        for i, variable in enumerate(self.signature.parameters):
            if i == 0 and variable != "x":
                raise ParameterValidationError("x must be the first arguments in your model equation.")
            elif i == 1 and variable != "y":
                raise ParameterValidationError("y must be the second arguments in your model equation")
            if i >= 2:
                self.parameters.append(variable.name)

    def __call__(self, *args, **kwargs) -> Real | np.ndarray:
        return self.function(*args, **kwargs)


class base_parameterization:
    def __init__(self, function: Callable):
        self.function = function
        self.signature: inspect.Signature = inspect.signature(function)

        if len(self.signature.parameters) < 2:
            raise ParameterValidationError(f"Found {len(self.signature.parameters)}")

        if len(self.signature.parameters) > 2:
            raise ParameterValidationError(f"Found function requiring {len(self.signature.parameters)} arguments."
                                           "Expected 2, only `x` and `y`.")

        for i, variable in enumerate(self.signature.parameters):
            if i == 0 and variable != "x":
                raise ParameterValidationError("x must be the first argument in your parameterization equation.")
            elif i == 1 and variable != "y":
                raise ParameterValidationError("y must be the second argument in your parameterization equation")

        origin_evaluation: dict[str, Any] = self.function(0, 0)
        self.parameters: list[str] = list(origin_evaluation.keys())

    def __call__(self, *args, **kwargs) -> dict[str, Any]:
        return self.function(*args, **kwargs)


class FunctionalModel:
    def __init__(self, equation: base_equation,
                 parameterization: base_parameterization | None):
        self._base_model: base_equation = equation
        self.variable = parameterization is not None
        self._parameterization: base_parameterization | None = parameterization

    def evaluate(self, x: Real | np.ndarray, y: Real | np.ndarray, size: int) -> EvaluatedModel:
        grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
        evaluations = dict()
        for xx in np.asarray(x):
            for yy in np.asarray(y):
                if self.variable:
                    evaluations[(xx, yy)] = self._base_model(grid_x, grid_y, **self._parameterization(x, y))
                else:
                    evaluations[(xx, yy)] = self._base_model(grid_x, grid_y)
        return EvaluatedModel(evaluations, size)


class EvaluatedModel:
    def __init__(self, evaluations: dict[Point, np.ndarray]):
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

    def correct_image(self, image: np.ndarray) -> np.ndarray:
        pass

    def at(self, xy: Point) -> np.ndarray:
        if xy in self._evaluation_points:
            return self._evaluations[xy]
        else:
            raise UnevaluatedPointError(f"Model not evaluated at {xy}.")

    def pad(self, new_size: int) -> EvaluatedModel:
        pass
