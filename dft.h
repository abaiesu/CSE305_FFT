#pragma once

#include "helpers.h"


void idft(CArray& y, IArray &dimensions, const int num_threads);
void serial_dft(CArray& y);
void parallel_dft(CArray &input, IArray &dimensions, int num_threads);
void serial_dft2D(TwoDCArray & data);
void gfft(CArray& x, double alpha);
void parallel_dft2D(TwoDCArray & data, int num_threads);
void inner_parallel_dft2D(TwoDCArray & data, IArray dimensions, int num_threads);
void flip(IArray& indices);
IArray get_ND_index(int index, IArray& dimensions);
int get_1D_index(IArray &coords, IArray &dimensions);