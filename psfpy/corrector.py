from __future__ import annotations

from typing import TypeAlias
import abc
from pathlib import Path
import warnings

import dill
import numpy as np
from spectrum import create_window
import deepdish as dd

from psfpy.exceptions import InvalidSizeError, EvaluatedModelInconsistentSizeError, UnevaluatedPointError
from psfpy.psf import VariedPSF, SimplePSF, PointSpreadFunctionABC

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

        if use_gpu:
            warnings.warn("The GPU acceleration is untested.", Warning)
            try:
                from cupy.fft import fft2, ifft2
            except ImportError as e:
                raise ImportError("cupy is required for GPU acceleration. cupy was not found.")
        else:
            try:
                from numpy.fft import fft2, ifft2
            except ImportError:
                raise ImportError("numpy is found for CPU execution. numpy was not found.")

        padding_shape = ((2*self._size, 2*self._size), (2*self._size, 2*self._size))
        padded_img = np.pad(image, padding_shape, mode="constant")
        result_img = np.zeros_like(padded_img)

        psf_target_padded = np.pad(self._target_evaluation, padding_shape, mode="constant")
        psf_target_hat = fft2(psf_target_padded)

        window1d = create_window(self._size, "cosine")
        apodization_window = np.sqrt(np.outer(window1d, window1d))

        for (x, y), this_psf_i in self._evaluations.items():
            this_psf_i_padded = np.pad(this_psf_i, padding_shape)
            this_psf_i_hat = fft2(this_psf_i_padded)
            this_psf_i_hat_abs = np.abs(this_psf_i_hat)
            this_psf_i_hat_norm = (np.conj(this_psf_i_hat) / this_psf_i_hat_abs) * (
                np.power(this_psf_i_hat_abs, alpha)
                / (np.power(this_psf_i_hat_abs, alpha + 1) + np.power(epsilon, alpha + 1))
            )

            img_i = get_padded_img_section(padded_img, x, y, self._size)
            img_i_apodized_padded = np.pad(img_i * apodization_window, padding_shape)
            img_i_hat = fft2(img_i_apodized_padded)

            corrected_i = np.real(ifft2(img_i_hat * this_psf_i_hat_norm * psf_target_hat))[
                self._size:self._size*2, self._size:self._size*2
            ]
            corrected_i = corrected_i * apodization_window
            set_padded_img_section(result_img, x, y, self._size, corrected_i)

        return result_img[
            self._size // 2: image.shape[0] + self._size // 2,
            self._size // 2: image.shape[1] + self._size // 2,
        ]

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


def get_padded_img_section(padded_img, x, y, psf_size) -> np.ndarray:
    """ Assumes an image is padded by ((psf_size, psf_size), (psf_size, psf_size))"""
    x_prime, y_prime = x + 2*psf_size, y + 2*psf_size
    return padded_img[x_prime: x_prime + psf_size, y_prime: y_prime + psf_size]


def set_padded_img_section(padded_img, x, y, psf_size, new_values) -> None:
    assert new_values.shape == (psf_size, psf_size)
    x_prime, y_prime = x + 2*psf_size, y + 2*psf_size
    padded_img[x_prime: x_prime + psf_size, y_prime: y_prime + psf_size] = new_values


# def calculate_covering(image_shape: tuple[int, int], size: int) -> np.ndarray:
#     half_size = np.ceil(size / 2).astype(int)
#
#     # x1, y1 are the primary grid. x2, y2 are the offset grid. Together they fully cover the image twice.
#     x1 = np.arange(-half_size, image_shape[0] + half_size, size)
#     y1 = np.arange(-half_size, image_shape[1] + half_size, size)
#     x1 = x1[x1 <= image_shape[0]+1]
#     y1 = y1[y1 <= image_shape[1]+1]
#
#     x2 = (x1 + half_size)[:-1]
#     y2 = (y1 + half_size)[:-1]
#
#     x1, y1 = np.meshgrid(x1, y1)
#     x2, y2 = np.meshgrid(x2, y2)
#
#     x1, y1 = x1.flatten(), y1.flatten()
#     x2, y2 = x2.flatten(), y2.flatten()
#
#     x = np.concatenate([x1, x2])
#     y = np.concatenate([y1, y2])
#     return np.stack([x, y], -1)

def calculate_covering(image_shape: tuple[int, int], size: int) -> np.ndarray:
    half_size = np.ceil(size / 2).astype(int)

    # x1, y1 are the primary grid. x2, y2 are the offset grid. Together they fully cover the image twice.
    x1 = np.arange(0, image_shape[0], size)
    y1 = np.arange(0, image_shape[1], size)

    x2 = np.arange(-half_size, image_shape[0]+half_size, size)
    y2 = np.arange(-half_size, image_shape[1]+half_size, size)

    x1, y1 = np.meshgrid(x1, y1)
    x2, y2 = np.meshgrid(x2, y2)

    x1, y1 = x1.flatten(), y1.flatten()
    x2, y2 = x2.flatten(), y2.flatten()

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    return np.stack([x, y], -1)