import numpy as np
cimport numpy as np
from cpython cimport array


cdef extern from "fft_speed.h":
    double * _correct_image(double *image, int image_width, int image_height,
                            int *x, int *y, int num_evaluations, int psf_size, double *evaluations,
                           double *target_psf, double alpha, double epsilon)

cpdef correct_image(image, x, y, evaluations, target_psf, alpha=0.5, epsilon=0.05):
    num_evaluations = evaluations.shape[0]
    psf_size = target_psf.shape[0]

    image_width, image_height = image.shape
    image_width_prime, image_height_prime = image_width + 2 * psf_size, image_height + 2 * psf_size

    cdef np.float_t[:] out = <np.float_t[:image_width_prime*image_height_prime]> _correct_image(array.array('d', image.flatten()).data.as_doubles,
                                                                                    image_width,
                                                                                    image_height,
                                                                                    array.array('i', x.flatten()).data.as_ints,
                                                                                    array.array('i', y.flatten()).data.as_ints,
                                                                                    num_evaluations,
                                                                                    psf_size,
                                                                                    array.array('d', evaluations.flatten()).data.as_doubles,
                                                                                    array.array('d', target_psf.flatten()).data.as_doubles,
                                                                                    alpha,
                                                                                    epsilon)
    arr = np.asarray(out).reshape((image_width+2*psf_size, image_height+2*psf_size))
    return arr[2*psf_size:image_width+2*psf_size, 2*psf_size:image_height+2*psf_size]