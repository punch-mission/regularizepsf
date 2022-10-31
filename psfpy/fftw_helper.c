// REALLY IMPORTANT TO INCLUDE complex.h before fftw3.h!!!
// forces it to use the standard C complex number definition
#include "fftw_helper.h"
#include<complex.h>
#include<fftw3.h>
#include<stdio.h>
#include<stdlib.h>

void setup(fft_data *data, int N) {
	//data = (fft_data *) malloc(sizeof(fft_data));
	data->N = N;
	data->data_in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N);
	data->data_out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N * N);
	// There are flags that allow you to change how much time
	// it takes for FFTs to occur at the cost of increased 'plan' creation time.
	// See documentation on FFTW_ESTIMATE, FFTW_MEASURE, etc.
	// Here, I'm just using the plan creator for 1D transforms.
	// This is about the simplest way to do it; there are much more complex
	// ways to create plans that allow you to perform the same transform
	// on multiple sets of data. I tend to use fftw_plan_dft because it works for
	// all ranks (1-D, 2-D, 3-D transforms, forwards and backwards), and
	// you can use it on different arrays rather than just the pointer y
	data->plan_forward = fftw_plan_dft_2d(N, N, data->data_in, data->data_out, -1, FFTW_MEASURE);
	data->plan_backward = fftw_plan_dft_2d(N, N, data->data_out, data->data_in, 1, FFTW_MEASURE);
	// If you're repeatedly doing transforms on the same size data in different
	// programme invocations, you can save the plan data to a file and load it
	// rather than recalculating it every time.
	// check out the documentation for:
	//  fftw_export_wisdom_to_filename
	//  fftw_import_wisdom_to_filename
}

void finalise(fft_data *data) {
	fftw_destroy_plan(data->plan_forward);
	fftw_destroy_plan(data->plan_backward);
	fftw_free(data->data_in);
	fftw_free(data->data_out);
	//free(data);
}

void execute_transform_forward(fft_data *data) {
	// If you only ever apply the FFT plan to
	// the array you created the plan with, you
	// can use the function fftw_execute(plan) here.
	// But sometimes you want to use it on other data which
	// isn't the same but is created in the same way, in
	// which case you use the function fftw_execute_dft
	fftw_execute(data->plan_forward);
}

void execute_transform_backward(fft_data *data) {
	fftw_execute(data->plan_backward);
}
