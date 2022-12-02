import numpy as np
from psfpy import simple_psf, varied_psf, FunctionalCorrector
from psfpy.corrector import calculate_covering


def elliptical_gaussian2d(x, y, height, cen_x, cen_y, sigma_x, sigma_y, rotation):
    a = np.square(np.cos(rotation)) / (2 * np.square(sigma_x)) + np.square(np.sin(rotation)) / (2 * np.square(sigma_y))
    b = - np.sin(2 * rotation) / (4 * np.square(sigma_x)) + np.sin(2 * rotation) / (4 * np.square(sigma_y))
    c = np.square(np.sin(rotation)) / (2 * np.square(sigma_x)) + np.square(np.cos(rotation)) / (2 * np.square(sigma_y))
    return height * np.exp(-(a * np.square(x - cen_x) + 2 * b * (x - cen_x) * (y - cen_y) + c * np.square(y - cen_y)))


@simple_psf
def elongated_gaussian(x, y,
                       core_height, core_x, core_y, core_sigma,
                       tail_height, tail_x, tail_y, tail_sigma_x, tail_sigma_y, tail_rotation,
                       background):
    core = elliptical_gaussian2d(x, y, core_height, core_x, core_y, core_sigma, core_sigma, 0)
    tail = elliptical_gaussian2d(x, y, tail_height, tail_x, tail_y, tail_sigma_x, tail_sigma_y, tail_rotation)
    return core + tail + background


@varied_psf(elongated_gaussian)
def punch_model(x, y):
    return {
        "core_height": 1,
        "core_x": 16,
        "core_y": 16,
        "core_sigma": (x+1)*3,
        "tail_height": (y+1)*4,
        "tail_x": 5 * np.cos(np.arctan2(y, x)),
        "tail_y": 5 * np.sin(np.arctan2(y, x)),
        "tail_sigma_x": 5,
        "tail_sigma_y": 1,
        "tail_rotation": np.arctan2(y, x)
    }


@simple_psf
def target_model(x, y):
    """ symmetric gaussian """
    x0, y0, sigma_x, sigma_y = 16, 16, 3, 3
    return np.exp(-(np.square(x-x0)/(2*np.square(sigma_x)) + np.square(y-y0)/(2*np.square(sigma_y))))


if __name__ == "__main__":
    uncorrected_image = np.zeros((1048, 1048))

    # use a functional model to correct an image
    my_model = FunctionalCorrector(punch_model, target_model)
    corrected_image = my_model.correct_image(uncorrected_image, 32)

    # save an evaluated array form of the model for faster computation in the future
    corners = calculate_covering(uncorrected_image.shape, 250)
    array_corrector = my_model.evaluate_to_array_form(corners[:, 0], corners[:, 1], 250)
    array_corrector.save("my_array_corrector.psfpy")
