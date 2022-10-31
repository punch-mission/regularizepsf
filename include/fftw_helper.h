#ifndef FFT_HELPER_H
#define FFT_HELPER_H
// REALLY IMPORTANT TO INCLUDE complex.h before fftw3.h!!!
// forces it to use the standard C complex number definition
#include<complex.h>
#include<fftw3.h>

typedef struct fft_data {
	int N;
	fftw_plan plan_forward;
	fftw_plan plan_backward;
	fftw_complex *data_in;
	fftw_complex *data_out;
} fft_data;

void setup(fft_data *data, int N);

void finalise(fft_data *data);

void execute_transform_forward(fft_data *data);

void execute_transform_backward(fft_data *data);

#endif