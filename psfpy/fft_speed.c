#include<complex.h>
#include<fftw3.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>


double ** pad(double **in, int in_width, int in_height, int pad_amount)
{
    int new_height = in_height + 2*pad_amount;
    int new_width = in_width + 2*pad_amount;
    double (*out)[new_width] = malloc(sizeof(double[new_width][new_height]));
    for (int i = 0; i < in_width; i++) {
        for (int j = 0; j < in_height; j++) {
            out[i + pad_amount][j + pad_amount] = in[i][j];
        }
    }
    return out;
}

//void depad(double *s, double *d, int dim)
//{
//    int i,j;
//    dim=dim-2;
//    for(i=0;i<dim;i++)
//        for(j=0;j<dim;j++)
//            *(d+i*dim+j)=*(s+(i*(dim+2)+(dim+3+j)));
//}

double square(double a) {
    return a * a;
}

double* _correct_image(double *image, int image_width, int image_height, int *x, int *y, int num_evaluations, int psf_size, double *evaluations, double *target_psf, double alpha, double epsilon) {
    // printf("inside c\n");
    fftw_complex *data_in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * psf_size * psf_size);
    fftw_complex *data_out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * psf_size * psf_size);

    fftw_plan plan = fftw_plan_dft_2d(psf_size, psf_size, data_in, data_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan reverse_plan = fftw_plan_dft_2d(psf_size, psf_size, data_out, data_in, FFTW_BACKWARD, FFTW_MEASURE);
    // printf("plan made\n");

    // build apodization window
    double * apodization_window = calloc(psf_size*psf_size, sizeof(double));
    for (int j = 0; j < psf_size; j++) {
        for (int k = 0; k < psf_size; k++) {
            apodization_window[j*psf_size + k] = square(sin((j + 0.5) * M_PI / psf_size)) * square(sin((k + 0.5) * M_PI / psf_size));
        }
    }
    int image_width_prime = image_width + 4 * psf_size;
    int image_height_prime = image_height + 4 * psf_size;

    double complex * psf_i_hat = calloc(psf_size*psf_size, sizeof(double complex));
    double complex * img_i_hat = calloc(psf_size*psf_size, sizeof(double complex));
    double complex * psf_i_hat_norm = calloc(psf_size*psf_size, sizeof(double complex));
    double complex * psf_target_hat = calloc(psf_size*psf_size, sizeof(double complex));
    double * result_image = calloc(image_width*image_height, sizeof(double));

    // calculate psf_target_hat
    for (int j = 0; j < psf_size; j++) {
        for (int k = 0; k < psf_size; k++) {
            data_in[j*psf_size + k] = target_psf[j*psf_size + k] + 0.0*I;
        }
    }
    fftw_execute(plan);
    for (int j=0; j < psf_size; j++) {
        for (int k = 0; k < psf_size; k++) {
            psf_target_hat[j*psf_size + k] = data_out[j*psf_size + k];
        }
    }

    // do computation for each region
    for (int i = 0; i < num_evaluations; i++) {
        int xx = x[i] + 2 * psf_size;
        int yy = y[i] + 2 * psf_size;

        // calculate psf_i_hat
        for (int j = 0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                data_in[j*psf_size + k] = evaluations[i*num_evaluations + j*psf_size + k] + 0.0*I;
            }
        }
        fftw_execute(plan);
        for (int j=0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                psf_i_hat[j*psf_size + k] = data_out[j*psf_size + k];
            }
        }

        // Calculate img_i_hat
        for (int j = 0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                if (0 <= xx + j && xx + j < image_width && 0 <= yy + k && yy + k < image_height) {
                    data_in[j*psf_size + k] = (image[(x[i] + j)*image_width + (y[i] + k)] * apodization_window[j*psf_size + k])+ 0.0 *I;
                } else {
                    data_in[j*psf_size + k] = 0.0 + 0.0 * I;
                }
            }
        }
        fftw_execute(plan);
        for (int j=0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                img_i_hat[j*psf_size + k] = data_out[j*psf_size + k];
            }
        }

        // Calculate corrected_i
        for (int j = 0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                double psf_i_hat_abs = cabs(psf_i_hat[j*psf_size + k]);
                psf_i_hat_norm[j * psf_size + k] = (conj(psf_i_hat[j* psf_size + k]) / psf_i_hat_abs) * (pow(psf_i_hat_abs, alpha) / (pow(psf_i_hat_abs, alpha + 1.0) + pow(epsilon * cabs(psf_target_hat[j * psf_size + k]), alpha + 1.0)));
                data_in[j*psf_size + k] = (img_i_hat[j*psf_size + k] * psf_i_hat_norm[j*psf_size + k] * psf_target_hat[j* psf_size + k]) + 0.0*I;
            }
        }
//        printf("in: %f\n", creal(data_in[0]));
        fftw_execute(reverse_plan);
//        printf("out: %f\n", creal(data_out[0]));

        for (int j=0; j < psf_size; j++) {
            for (int k = 0; k < psf_size; k++) {
                // printf("%f\n", creal(data_out[j*psf_size + k]));
                if (0 <= xx + j && xx + j < image_width && 0 <= yy + k && yy + k < image_height) {
                    result_image[(xx+j)*psf_size + (yy+k)] = result_image[(xx+j)*psf_size + (yy+k)] + (creal(data_out[j*psf_size + k]) * apodization_window[j*psf_size + k]);
                    result_image[(x[i]+j)*image_width + y[i] + k] = 1.0;
                }
            }
        }
    }

    free(psf_i_hat);
    free(img_i_hat);
    free(psf_i_hat_norm);
    free(psf_target_hat);
    free(data_in);
    free(data_out);

    return result_image;
}

int main(int argc, char *argv[]) {
    int width = 2;
    int height = 3;
    int pad_amount = 2;

    double (*img)[width] = malloc(sizeof(double[width][height]));

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            img[i][j] = 2.0;
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%f ", img[i][j]);
        }
        printf("\n");
    }


    printf("\n");
    double ** padded_img = pad(img, width, height, pad_amount);

    for (int i = 0; i < width + 2*pad_amount; i++) {
        for (int j = 0; j < height + 2*pad_amount; j++) {
            printf("%f ", padded_img[i][j]);
        }
        printf("\n");
    }

    free(img);
    free(padded_img);


    int num_evaluations = 100;
    int psf_size = 70;
//
//
//    double *image = calloc(2048*2048, sizeof(double));
//    int *x = calloc(num_evaluations, sizeof(int));
//    int *y = calloc(num_evaluations, sizeof(int));
//    double *evaluations = calloc(num_evaluations*psf_size*psf_size, sizeof(double));
//    double *target_psf = calloc(psf_size * psf_size, sizeof(double));
//    double alpha = 0.5;
//    double epsilon = 0.05;
//
//    for (int i = 0; i < num_evaluations; i++) {
//        x[i] = i;
//        y[i] = i;
//        for (int j = 0; j < psf_size; j++) {
//            for (int k = 0; k < psf_size; k++) {
//                evaluations[i*num_evaluations+j*psf_size+k] = 1.0;
//                target_psf[j*psf_size+k] = 1.0;
//            }
//        }
//    }
//
//    for (int i = 0; i < 2048; i++) {
//        for (int j = 0; j < 2048; j++) {
//            image[i*2048+j] = 2.0;
//        }
//    }
//    clock_t start, end;
//    double cpu_time_used;
//    start = clock();
//
//     double * output = _correct_image(image, 2048, 2048, x, y, num_evaluations, psf_size, evaluations, target_psf, alpha, epsilon);
//     // printf("output[0][0] = %f\n", output[0]);
////     for (int i; i <= 4787344; i++) {
////        if (output[i] != 0.0  && !(output[i] != output[i])) {
////            printf("%d %f\n", i, *(output + i));
////        }
////     }
//     end = clock();
//     cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//     printf("%f \n", cpu_time_used);
}