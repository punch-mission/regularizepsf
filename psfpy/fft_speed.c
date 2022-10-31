#include<complex.h>
#include<fftw3.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>


double square(double a) {
    return a * a;
}

double* _correct_image(double *image, int image_width, int image_height, int *x, int *y, int num_evaluations, int psf_size, double *evaluations, double *target_psf, double alpha, double epsilon) {
    fftw_complex *data_in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * psf_size * psf_size);
    fftw_complex *data_out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * psf_size * psf_size);

    fftw_plan plan = fftw_plan_dft_2d(psf_size, psf_size, data_in, data_out, -1, FFTW_MEASURE);

    // build apodization window
    double apodization_window[psf_size][psf_size];
    for (int j = 0; j < psf_size; j++) {
        for (int k = 0; k < psf_size; k++) {
            apodization_window[j][k] = square(sin((j + 0.5) * M_PI / psf_size)) * square(sin((k + 0.5) * M_PI / psf_size));
        }
    }

    double complex psf_i_hat[psf_size][psf_size];
    double complex img_i_hat[psf_size][psf_size];
    double complex psf_i_hat_norm[psf_size][psf_size];
    double complex psf_target_hat[psf_size][psf_size];
    double * result_image = calloc((image_width+(2*psf_size))*(image_height+(2*psf_size)), sizeof(double));
    for (int i = 0; i < num_evaluations; i++) {
        int xx = x[i] + 2 * psf_size;
        int yy = y[i] + 2 * psf_size;

        // Calculate psf_i_hat
        for (int j = 0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                data_in[j*psf_size + k] = evaluations[i*num_evaluations + j*psf_size + k*psf_size];
            }
        }
        fftw_execute(plan);
        for (int j=0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                psf_i_hat[j][k] = data_out[j*psf_size + k];
            }
        }

        // Calculate img_i_hat
        for (int j = 0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                if (0 <= xx + j && xx + j < image_width && 0 <= yy + k && yy + k < image_height) {
                    data_in[j*psf_size + k] = image[(xx + j)*image_width + (yy + k)];
                } else {
                    data_in[j*psf_size + k] = 0.0 + 0.0 * I;
                }
            }
        }
        fftw_execute(plan);
        for (int j=0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                psf_i_hat[j][k] = data_out[j*psf_size + k];
            }
        }

        // Calculate corrected_i
        for (int j = 0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                double psf_i_hat_abs = cabs(psf_i_hat[j][k]);
                psf_i_hat_norm[j][k] = (conj(psf_i_hat[j][k]) / psf_i_hat_abs) * (pow(psf_i_hat_abs, alpha) / (pow(psf_i_hat_abs, alpha + 1.0) + pow(epsilon * cabs(psf_target_hat[j][k]), alpha + 1.0)));
                data_in[j*psf_size + k] = img_i_hat[j][k] * psf_i_hat_norm[j][k] * psf_target_hat[j][k];
            }
        }
        fftw_execute(plan);
        for (int j=0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                result_image[(xx+j)*psf_size + (yy+k)] = result_image[(xx+j)*psf_size + (yy+k)] + (creal(data_out[j*psf_size + k]) * apodization_window[j][k]);
            }
        }
    }
    return result_image;
}
