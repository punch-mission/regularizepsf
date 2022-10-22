import numpy as np

import psfpy


@psfpy.base_equation
def gaussian(x, y, sigma):
    pass


@psfpy.base_parameterization
def psf_variability(x, y):
    return {"sigma": 0.5}


uncorrected_image = np.zeros((100, 100))


functional_model = psfpy.FunctionalModel(gaussian, psf_variability)
evaluated_model = functional_model.evaluate(np.arange(100), np.arange(100), 50)
corrected_image = evaluated_model.correct_image(uncorrected_image)


# from typing import Dict, Tuple
# from datetime import datetime
#
# from prefect import task, get_run_logger
# import numpy as np
# from numpy.fft import fft2, ifft2
# from spectrum import create_window
#
# from punchbowl.data import PUNCHData
#
#
# def get_padded_img_section(padded_img, x, y, psf_size) -> np.ndarray:
#     x_prime, y_prime = x + psf_size // 2, y + psf_size // 2
#     return padded_img[x_prime : x_prime + psf_size, y_prime : y_prime + psf_size]
#
#
# def set_padded_img_section(padded_img, x, y, psf_size, new_values) -> None:
#     assert new_values.shape == (psf_size, psf_size)
#     x_prime, y_prime = x + psf_size // 2, y + psf_size // 2
#     padded_img[x_prime : x_prime + psf_size, y_prime : y_prime + psf_size] = new_values
#
#
# def correct_psf(
#     img: np.ndarray,
#     psf_i: Dict[Tuple[int, int], np.ndarray],
#     psf_target: np.ndarray,
#     alpha: float = 0,
#     epsilon: float = 0.035,
# ) -> np.ndarray:
#     assert len(img.shape) == 2, "img must be a 2-d numpy array."
#     psf_i_shape = next(iter(psf_i.values())).shape
#     assert len(psf_i_shape) == 2, "psf_i entries must be 2-d numpy arrays."
#     assert psf_i_shape[0] == psf_i_shape[1], "PSFs must be square"
#     assert psf_i_shape[0] % 2 == 0, "PSF size must be even in both dimensions"
#     assert all(
#         v.shape == psf_i_shape for v in psf_i.values()
#     ), "All psf_i entries must be the same shape."
#     assert (
#         psf_target.shape == psf_i_shape
#     ), "Shapes of psf_i and psf_target do not match."
#     assert all(
#         img_dim_i >= psf_dim_i for img_dim_i, psf_dim_i in zip(img.shape, psf_i_shape)
#     ), "img must be at least as large as the PSFs in all dimensions"
#
#     psf_size = psf_i_shape[0]
#     padding_shape = ((psf_size // 2, psf_size // 2), (psf_size // 2, psf_size // 2))
#     padded_img = np.pad(img, padding_shape, mode="constant")
#     result_img = np.zeros_like(padded_img)
#
#     psf_target_hat = fft2(psf_target)
#
#     window1d = create_window(psf_size, "cosine")
#     apodization_window = np.sqrt(np.outer(window1d, window1d))
#
#     for (x, y), this_psf_i in psf_i.items():
#         this_psf_i_padded = np.pad(this_psf_i, padding_shape)
#         this_psf_i_hat = fft2(this_psf_i_padded)
#         this_psf_i_hat_abs = np.abs(this_psf_i_hat)
#         this_psf_i_hat_norm = (np.conj(this_psf_i_hat) / this_psf_i_hat_abs) * (
#             np.power(this_psf_i_hat_abs, alpha)
#             / (np.power(this_psf_i_hat_abs, alpha + 1) + np.power(epsilon, alpha + 1))
#         )
#
#         img_i = get_padded_img_section(padded_img, x, y, psf_size)
#         img_i_apodized_padded = np.pad(img_i * apodization_window, padding_shape)
#         img_i_hat = fft2(img_i_apodized_padded)
#
#         corrected_i = np.real(ifft2(img_i_hat * this_psf_i_hat_norm * psf_target_hat))[
#             psf_size:psf_size:
#         ]
#         corrected_i = corrected_i * apodization_window
#         set_padded_img_section(result_img, x, y, psf_size, corrected_i)
#     return result_img[
#         psf_size // 2 : img.shape[0] + psf_size // 2,
#         psf_size // 2 : img.shape[1] + psf_size // 2,
#     ]
#
#
# @task
# def correct_psf_task(data_object: PUNCHData) -> PUNCHData:
#     logger = get_run_logger()
#     logger.info("correct_psf started")
#     # TODO: do PSF correction in here
#     logger.info("correct_psf finished")
#     data_object.add_history(datetime.now(), "LEVEL1-correct_psf", "PSF corrected")
#     return data_object
