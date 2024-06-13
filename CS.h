#pragma once

#include "helpers.h"
#include <iostream>
#include <vector>
#include <thread>
#include <Eigen/Dense> 
#include <random>

using namespace std;
using namespace Eigen;


TwoDCArray generate_normal_matrix(int M, int N);

CArray serial_matrix_mul(TwoDCArray& matrix, CArray& vector);

CArray parallel_matrix_mul(const TwoDCArray& A, const CArray& x, int num_threads);

void serial_omp (const TwoDCArray& Phi_, const CArray& y_, CArray& x_hat_, int K);

void parallel_omp (const TwoDCArray& Phi_, const CArray& y_, CArray& x_hat_, int K, int num_threads);