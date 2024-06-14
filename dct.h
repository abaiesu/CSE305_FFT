#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <thread>
#include <chrono>
#include "dft.h"
#include "helpers.h"

DArray serial_dct(const DArray& input);

DArray parallel_dct(const DArray& input, IArray dimensions, int num_threads);

DArray serial_idct(const DArray& input);

DArray parallel_idct(DArray& input, IArray dimensions, int num_threads);
