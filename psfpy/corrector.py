from __future__ import annotations

import os
from typing import TypeAlias
import abc
from pathlib import Path
import warnings
from multiprocessing import Process, Semaphore, Lock, Pool

import dill
import numpy as np
from spectrum import create_window
import deepdish as dd
from numpy.fft import fft2, ifft2

from psfpy.exceptions import InvalidSizeError, EvaluatedModelInconsistentSizeError, UnevaluatedPointError
from psfpy.psf import VariedPSF, SimplePSF, PointSpreadFunctionABC
from psfpy.helper import _correct_image

Point: TypeAlias = tuple[int, int]


class CorrectorABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the model to a file.

        Parameters
        ----------
        path : str or `pathlib.Path`
            where to save the model, suggested extension is ".corr"

        Returns
        -------
        None
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path) -> CorrectorABC:
        """Loads a model from the path

        Parameters
        ----------
        path

        Returns
        -------

        """

    @abc.abstractmethod
    def correct_image(self, image: np.ndarray, size: int = None,
                      alpha: float = 0.5, epsilon: float = 0.05, use_gpu: bool = False) -> np.ndarray:
        """

        Parameters
        ----------
        image
        size
        alpha
        epsilon
        use_gpu

        Returns
        -------

        """


class FunctionalCorrector(CorrectorABC):
    def __init__(self, psf: PointSpreadFunctionABC,
                 target_model: SimplePSF | None):
        self._psf: PointSpreadFunctionABC = psf
        self._variable: bool = isinstance(self._psf, VariedPSF)
        self._target_model: SimplePSF = target_model

    @property
    def variable(self) -> bool:
        return self._variable

    def evaluate_to_array_form(self, x: np.ndarray, y: np.ndarray, size: int) -> ArrayCorrector:
        if size % 2 != 0:
            raise InvalidSizeError(f"size must be even. Found size={size}.")

        image_x, image_y = np.meshgrid(np.arange(size), np.arange(size))
        evaluations = dict()
        for xx in x:
            for yy in y:
                evaluations[(xx, yy)] = self._psf(image_x, image_y)

        target_evaluation = self._target_model(image_x, image_y)
        return ArrayCorrector(evaluations, target_evaluation)

    def correct_image(self, image: np.ndarray, size: int = None,
                      alpha: float = 0.5, epsilon: float = 0.05, use_gpu: bool = False) -> np.ndarray:
        corners = calculate_covering(image.shape, size)
        array_corrector = self.evaluate_to_array_form(corners[:, 0], corners[:, 1], size)
        return array_corrector.correct_image(image, size=size, alpha=alpha, epsilon=epsilon, use_gpu=use_gpu)

    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return dill.load(f)


class ArrayCorrector(CorrectorABC):
    def __init__(self, evaluations: dict[Point, np.ndarray], target_evaluation: np.ndarray):
        self._evaluation_points: list[Point] = list(evaluations.keys())

        if len(evaluations[self._evaluation_points[0]].shape) != 2:
            raise InvalidSizeError(f"PSF evaluations must be 2-D numpy arrays.")
        self._size = evaluations[self._evaluation_points[0]].shape[0]
        if self._size <= 0:
            raise InvalidSizeError(f"Found size of {self._size}. Must be >= 1")
        if self._size % 2 != 0:
            raise InvalidSizeError(f"Size must be even. Found {self._size}")

        self._evaluations: dict[Point, np.ndarray] = evaluations
        for (x, y), evaluation in self._evaluations.items():
            if evaluation.shape != (self._size, self._size):
                raise EvaluatedModelInconsistentSizeError(f"Expected evaluated model to have shapes of "
                                                          f"{(self._size, self._size)}. Found {evaluation.shape} "
                                                          f"at {(x, y)}.")

        self._target_evaluation = target_evaluation
        if self._target_evaluation.shape != (self._size, self._size):
            raise EvaluatedModelInconsistentSizeError("The target and evaluations must have the same shape.")

    def correct_image(self, image: np.ndarray, size: int = None,
                      alpha: float = 0.5, epsilon: float = 0.05, use_gpu: bool = False) -> np.ndarray:
        if not all(img_dim_i >= psf_dim_i for img_dim_i, psf_dim_i in zip(image.shape, (self._size, self._size))):
            raise InvalidSizeError("The image must be at least as large as the PSFs in all dimensions")
        return _correct_image(image, self._size, self._target_evaluation, self._evaluations, alpha, epsilon)

    def __getitem__(self, xy: Point) -> np.ndarray:
        if xy in self._evaluation_points:
            return self._evaluations[xy]
        else:
            raise UnevaluatedPointError(f"Model not evaluated at {xy}.")

    def save(self, path):
        dd.io.save(path, (self._evaluations, self._target_evaluation))

    @classmethod
    def load(cls, path):
        evaluations, target_evaluation = dd.io.load(path)
        return cls(evaluations, target_evaluation)


def calculate_covering(image_shape: tuple[int, int], size: int) -> np.ndarray:
    half_size = np.ceil(size / 2).astype(int)

    x1 = np.arange(0, image_shape[0], size)
    y1 = np.arange(0, image_shape[1], size)

    x2 = np.arange(-half_size, image_shape[0]+half_size, size)
    y2 = np.arange(-half_size, image_shape[1]+half_size, size)

    x3 = np.arange(-half_size, image_shape[0]+half_size, size)
    y3 = np.arange(0, image_shape[1], size)

    x4 = np.arange(0, image_shape[0], size)
    y4 = np.arange(-half_size, image_shape[1]+half_size, size)

    x1, y1 = np.meshgrid(x1, y1)
    x2, y2 = np.meshgrid(x2, y2)
    x3, y3 = np.meshgrid(x3, y3)
    x4, y4 = np.meshgrid(x4, y4)

    x1, y1 = x1.flatten(), y1.flatten()
    x2, y2 = x2.flatten(), y2.flatten()
    x3, y3 = x3.flatten(), y3.flatten()
    x4, y4 = x4.flatten(), y4.flatten()

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    return np.stack([x, y], -1)
