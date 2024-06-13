#include "helpers.h"


void ifft(CArray& y, IArray &dimensions, const int num_threads);
void sequ_fft(CArray& y);
void parallel_fft(CArray &input, IArray &dimensions, int num_threads);
void fft2D(TwoDCArray & data, IArray &dimensions, int num_threads);