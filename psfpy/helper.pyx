import numpy as np
from numpy.fft import fft2, ifft2

def get_padded_img_section(padded_img, x, y, psf_size) -> np.ndarray:
    """ Assumes an image is padded by ((2*psf_size, 2*psf_size), (2*psf_size, 2*psf_size))"""
    x_prime, y_prime = x + 2*psf_size, y + 2*psf_size
    return padded_img[x_prime: x_prime + psf_size, y_prime: y_prime + psf_size]


def set_padded_img_section(padded_img, x, y, psf_size, new_values) -> None:
    assert new_values.shape == (psf_size, psf_size)
    x_prime, y_prime = x + 2*psf_size, y + 2*psf_size
    padded_img[x_prime: x_prime + psf_size, y_prime: y_prime + psf_size] = new_values


def add_padded_img_section(padded_img, x, y, psf_size, new_values) -> None:
    assert new_values.shape == (psf_size, psf_size)
    prior = get_padded_img_section(padded_img, x, y, psf_size)
    set_padded_img_section(padded_img, x, y, psf_size, new_values + prior)


def _correct_image(image, size, target_evaluation, evaluations, alpha, epsilon):
    padding_shape = ((2 * size, 2 * size), (2 * size, 2 * size))
    padded_img = np.pad(image, padding_shape, mode="constant")
    result_img = np.zeros_like(padded_img)

    psf_target_padded = target_evaluation
    psf_target_hat = fft2(psf_target_padded)

    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    apodization_window = np.square(np.sin((xx + 0.5) * np.pi / size)) * np.square(
        np.sin((yy + 0.5) * np.pi / size))

    for (x, y), psf_i in evaluations.items():
        img_i = get_padded_img_section(padded_img, x, y, size)

        corrected_i = _correct_patch(img_i, psf_i, psf_target_hat, apodization_window, alpha, epsilon)

        add_padded_img_section(result_img, x, y, size, corrected_i)

    return result_img[2 * size:image.shape[0] + 2 * size,
           2 * size:image.shape[1] + 2 * size]

def _correct_patch(img_i, psf_i, psf_target_hat, apodization_window, alpha, epsilon):
    psf_i_hat = fft2(psf_i)
    psf_i_hat_abs = np.abs(psf_i_hat)
    psf_i_hat_norm = (np.conj(psf_i_hat) / psf_i_hat_abs) * (
            np.power(psf_i_hat_abs, alpha)
            / (np.power(psf_i_hat_abs, alpha + 1) + np.power(epsilon * np.abs(psf_target_hat), alpha + 1))
    )

    img_i_apodized = img_i * apodization_window
    img_i_hat = fft2(img_i_apodized)

    corrected_i = np.real(ifft2(img_i_hat * psf_i_hat_norm * psf_target_hat))
    corrected_i = corrected_i * apodization_window

    return corrected_i