#pragma once

#include "helpers.h"
#include <iostream>
#include <vector>
#include <thread>
#include <Eigen/Dense> 
#include <random>

using namespace std;
using namespace Eigen;


TwoDDArray generate_normal_matrix(int M, int N);

DArray serial_matrix_vect_mul(TwoDDArray& matrix, DArray& vector);

DArray parallel_matrix_mul(const TwoDDArray& A, const DArray& x, int num_threads);

void serial_omp (const TwoDDArray& Phi_, const DArray& y_, DArray& x_hat_, int K);

void parallel_omp (const TwoDDArray& Phi_, const DArray& y_, DArray& x_hat_, int K, int num_threads);

