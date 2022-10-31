import numpy as np
cimport numpy as np
from cpython cimport array
from ctypes import c_void_p, c_double, c_int, cdll
from numpy.ctypeslib import ndpointer



cdef extern from "fft_speed.h":
    double * _correct_image(double *image, int image_width, int image_height,
                            int *x, int *y, int num_evaluations, int psf_size, double *evaluations,
                           double *target_psf, double alpha, double epsilon)



cpdef c_correct_image(image, x, y, evaluations, target_psf, alpha, epsilon):
    num_evaluations = evaluations.shape[0]
    psf_size = target_psf.shape[0]
    image_width, image_height = image.shape
    image_width_prime, image_height_prime = int(image_width + 2 * psf_size), int(image_height + 2 * psf_size)
    # cdef np.ndarray[np.double_t, ndim=1, mode = 'c'] np_buff = np.ascontiguousarray(image.flatten(), dtype=np.float64)
    # cdef double * im_buff = <double *> np_buff.data
    #
    # cdef np.ndarray[np.double_t, ndim=1, mode = 'c'] evaluations_buff = np.ascontiguousarray(evaluations.flatten(), dtype=np.float64)
    # cdef double * evaluations_ptr = <double *> evaluations_buff.data

    # cdef array.array image_arr = array.array('d', image.flatten())
    # cdef array.array evaluations_arr = array.array('d', evaluations.flatten())

    # lib = cdll.LoadLibrary("psfpy/fft_speed.so")
    # _correct_image = lib._correct_image  #c_sum is the name of our C function
    # _correct_image.restype = ndpointer(dtype=c_double, shape=(np.prod(image.shape),))
    #
    # out = _correct_image(c_void_p(image.flatten().ctypes.data),
    #                c_int(image_width), c_int(image_height),
    #                c_void_p(x.flatten().ctypes.data),
    #                c_void_p(y.flatten().ctypes.data),
    #                c_int(num_evaluations), c_int(psf_size),
    #                c_void_p(evaluations.flatten().ctypes.data),
    #                c_void_p(target_psf.flatten().ctypes.data),
    #                c_double(alpha), c_double(epsilon)
    #                )
    print("all zeros in cython?", np.all(image == 0))
    cdef np.float_t[:] out = <np.float_t[:image_width_prime*image_height_prime]> _correct_image(
                                                                                    array.array('d', image.flatten()).data.as_doubles,
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
    print("all zeros after?", np.all(np.asarray(out)==0))
    return np.asarray(out)
    # arr = np.asarray(out).reshape((image_width+2*psf_size, image_height+2*psf_size))
    # return arr[2*psf_size:image_width+2*psf_size, 2*psf_size:image_height+2*psf_size]