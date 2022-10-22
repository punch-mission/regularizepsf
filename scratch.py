import numpy as np

import psfpy


@psfpy.base_equation
def gaussian(x, y, x0, y0, sigma_x, sigma_y):
    return np.exp(-(np.square(x-x0)/(2*np.square(sigma_x)) + np.square(y-y0)/(2*np.square(sigma_y))))


@psfpy.base_equation
def target_model(x, y):
    return gaussian(x, y, 16, 16, 3, 3)


@psfpy.base_parameterization(reference_function=gaussian)
def psf_variability(x, y):
    return {
        "x0": 16,
        "y0": 16,
        "sigma_x": 4.6,
        "sigma_y": 3.3
    }


uncorrected_image = np.zeros((100, 100))


functional_model = psfpy.FunctionalModel(gaussian, psf_variability, target_model)
evaluated_model = functional_model.evaluate(np.arange(100), np.arange(100), 100)
print(evaluated_model[0, 0])
# corrected_image = evaluated_model.correct_image(uncorrected_image)
