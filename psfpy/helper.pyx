# cython: linetrace = True

import numpy as np
cimport numpy as np
from numpy.fft import fft2, ifft2
from cython.parallel import prange
cimport cython

ctypedef np.float_t DTYPE_t

cdef extern from "fftw3.h":
    ctypedef double fftw_complex[2]

# Because we did the 'complex.h' import
# we know the data types complex and fftw_complex
# are compatible so this is fine!
cdef extern from "fftw_helper.h":
    struct fft_data:
        complex *data_in;
        complex *data_out;
    void setup(fft_data *data, int N)
    void finalise(fft_data *data)
    void execute_transform_forward(fft_data *data)
    void execute_transform_backward(fft_data *data)

cdef class FFTObject(object):
    cdef fft_data data
    cdef public int N
    cdef np.complex128_t[:, :] data_in_ptr;
    cdef np.complex128_t[:, :] data_out_ptr;
    cdef public np.ndarray data_in, data_out
    def __cinit__(self, N):
        self.N = N
        setup(&self.data, N)
        self.data_in_ptr = <np.complex128_t[:self.N, :self.N]> self.data.data_in
        self.data_out_ptr = <np.complex128_t[:self.N, :self.N]> self.data.data_out
        self.data_in = np.asarray(self.data_in_ptr)
        self.data_out = np.asarray(self.data_out_ptr)

    def __dealloc__(self):
        finalise(&self.data)

    def fft(self):
        execute_transform_forward(&self.data)

    def ifft(self):
        execute_transform_backward(&self.data)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef _correct_image(np.ndarray[DTYPE_t, ndim=2] image,
                     np.ndarray[DTYPE_t, ndim=2] target_evaluation,
                     np.ndarray[np.int_t, ndim=1] x,
                     np.ndarray[np.int_t, ndim=1] y,
                     np.ndarray[DTYPE_t, ndim=3] values,
                     float alpha,
                     float epsilon):
    cdef int num_evaluations = x.shape[0]
    cdef int size = values.shape[1]
    cdef int i = 0, j=0, xx = 0, yy = 0
    cdef int this_x, this_y, this_x_prime, this_y_prime
    cdef np.ndarray[DTYPE_t, ndim=2] img_i = np.empty((size, size), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] psf_i = np.empty((size, size), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] corrected_i = np.empty((size, size), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] padded_img = np.pad(image, ((2 * size, 2* size), (2*size, 2*size)),
                                                         mode="constant")
    cdef np.ndarray[DTYPE_t, ndim=2] result_img = np.zeros_like(padded_img)
    cdef np.ndarray[np.complex128_t, ndim=2] psf_target_hat = fft2(target_evaluation)
    cdef np.ndarray[np.complex128_t, ndim=2] psf_i_hat = np.empty((size, size), dtype=np.complex)
    cdef complex psf_i_hat_abs
    cdef np.ndarray[np.complex128_t, ndim=2] psf_i_hat_norm = np.empty((size, size), dtype=np.complex)
    cdef np.ndarray[np.complex128_t, ndim=2] img_i_hat = np.empty((size, size), dtype=np.complex)
    cdef np.ndarray[np.complex128_t, ndim=2] temp = np.empty((size, size), dtype=np.complex)
    cdef np.ndarray[np.float_t, ndim=2] apodization_window
    # cdef fftwnd_plan plan = fftw2d_create_plan(size, size, FFTW_FORWARD, FFTW_ESTIMATE)
    cdef fft_data data

    xarr, yarr = np.meshgrid(np.arange(size), np.arange(size))
    apodization_window = np.square(np.sin((xarr + 0.5) * np.pi / size)) * np.square(
        np.sin((yarr + 0.5) * np.pi / size))

    fft_handler = FFTObject(size)

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
                fft_handler.data_in[xx, yy] = psf_i[xx, yy]
        fft_handler.fft()
        for xx in range(size):
            for yy in range(size):
                psf_i_hat[xx, yy] = fft_handler.data_out[xx, yy]

        # psf_i_hat = fft2(psf_i)

        for xx in range(size):
            for yy in range(size):
                img_i[xx, yy] = apodization_window[xx, yy] * padded_img[this_x_prime + xx, this_y_prime + yy]
                fft_handler.data_in[xx, yy] = img_i[xx, yy]
        fft_handler.fft()
        for xx in range(size):
            for yy in range(size):
                img_i_hat[xx, yy] = fft_handler.data_out[xx, yy]

        # img_i_hat = fft2(img_i)

        for xx in range(size):
            for yy in range(size):
                psf_i_hat_abs = abs(psf_i_hat[xx, yy])
                psf_i_hat_norm[xx, yy] = (psf_i_hat[xx, yy].conjugate() / psf_i_hat_abs) * ((psf_i_hat_abs**alpha) / ((psf_i_hat_abs**(alpha+1.0)) + (epsilon * abs(psf_target_hat[xx, yy]))**(alpha+1.0)))
                temp[xx, yy] = img_i_hat[xx, yy] * psf_i_hat_norm[xx, yy] * psf_target_hat[xx, yy]
                fft_handler.data_out[xx, yy] = temp[xx, yy]
        fft_handler.ifft()
        for xx in range(size):
            for yy in range(size):
                temp[xx, yy] = fft_handler.data_in[xx, yy]
        # temp = ifft2(temp)

        # add the corrected_i to the array
        for xx in range(size):
            for yy in range(size):
                result_img[this_x_prime+xx, this_y_prime+yy] = result_img[this_x_prime+xx, this_y_prime+yy] + (temp[xx, yy].real * apodization_window[xx,yy])

    return result_img[2 * size:image.shape[0] + 2 * size, 2 * size:image.shape[1] + 2 * size]
