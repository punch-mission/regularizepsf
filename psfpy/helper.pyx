import numpy as np
cimport numpy as np
from numpy.fft import fft2, ifft2
cimport cython

ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _correct_image(np.ndarray[np.float_t, ndim=2] image,
                     np.ndarray[DTYPE_t, ndim=2] target_evaluation,
                     np.ndarray[np.int_t, ndim=1] x,
                     np.ndarray[np.int_t, ndim=1] y,
                     np.ndarray[DTYPE_t, ndim=3] values,
                     float alpha,
                     float epsilon):
    """Core Cython code to actually correct an image"""
    cdef int num_evaluations = x.shape[0]
    cdef int size = values.shape[1]
    cdef int i = 0, j=0, xx = 0, yy = 0
    cdef int this_x, this_y, this_x_prime, this_y_prime
    cdef np.ndarray[DTYPE_t, ndim=2] img_i = np.empty((size, size), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] psf_i = np.empty((size, size), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] corrected_i = np.empty((size, size), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] padded_img = np.pad(image, ((2 * size, 2* size), (2*size, 2*size)),
                                                         mode="constant")
    cdef np.ndarray[DTYPE_t, ndim=2] result_img = np.zeros_like(padded_img, dtype=np.float)
    cdef np.ndarray[np.complex128_t, ndim=2] psf_target_hat = fft2(target_evaluation)
    cdef np.ndarray[np.complex128_t, ndim=2] psf_i_hat = np.empty((size, size), dtype=np.complex)
    cdef float psf_i_hat_abs
    cdef np.ndarray[np.complex128_t, ndim=2] psf_i_hat_norm = np.empty((size, size), dtype=np.complex)
    cdef np.ndarray[np.complex128_t, ndim=2] img_i_hat = np.empty((size, size), dtype=np.complex)
    cdef np.ndarray[np.complex128_t, ndim=2] temp = np.empty((size, size), dtype=np.complex)
    cdef np.ndarray[np.float_t, ndim=2] apodization_window


    xarr, yarr = np.meshgrid(np.arange(size), np.arange(size))
    apodization_window = np.square(np.sin((xarr + 0.5) * (np.pi / size))) * np.square(np.sin((yarr + 0.5) * (np.pi / size)))
    apodization_window = np.sin((xarr + 0.5) * (np.pi / size)) * np.sin((yarr + 0.5) * (np.pi / size))

    for i in range(num_evaluations):
        # get the x and the y
        this_x = x[i]
        this_y = y[i]
        this_x_prime = this_x + 2 * size
        this_y_prime = this_y + 2 * size

        # copy the apodized img_i from the padded image for processing
        for xx in range(size):
            for yy in range(size):
                psf_i[xx, yy] = values[i, xx, yy]
        psf_i_hat = fft2(psf_i)

        for xx in range(size):
            for yy in range(size):
                img_i[xx, yy] = apodization_window[xx, yy] * padded_img[this_x_prime + xx, this_y_prime + yy]
        img_i_hat = fft2(img_i)

        for xx in range(size):
            for yy in range(size):
                psf_i_hat_abs = abs(psf_i_hat[xx, yy])
                psf_i_hat_norm[xx, yy] = (psf_i_hat[xx, yy].conjugate() / psf_i_hat_abs) * ((psf_i_hat_abs**alpha) / ((psf_i_hat_abs**(alpha+1.0)) + (epsilon * abs(psf_target_hat[xx, yy]))**(alpha+1.0)))
                temp[xx, yy] = img_i_hat[xx, yy] * psf_i_hat_norm[xx, yy] * psf_target_hat[xx, yy]
        corrected_i = np.real(ifft2(temp))

        # add the corrected_i to the array
        for xx in range(size):
            for yy in range(size):
                result_img[this_x_prime+xx, this_y_prime+yy] = result_img[this_x_prime+xx, this_y_prime+yy] + (corrected_i[xx, yy] * apodization_window[xx, yy])

    return result_img[2 * size:image.shape[0] + 2 * size, 2 * size:image.shape[1] + 2 * size]
