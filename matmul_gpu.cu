#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <thread>
#include <algorithm>
#include "helpers.h"


// CUDA kernel for matrix-vector multiplication
__global__ void matrixVectorGPU(const double* matrix, const double* vector, double* result, int N) {
   
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        double sum = 0;
        for (int j = 0; j < N; ++j) {
            sum += matrix[row * N + j] * vector[j];
        }
        result[row] = sum;
    }
}


void matrix_vect_mul_GPU(const TwoDDArray& matrix, const DArray& vector, DArray& resultGPU, double& GPU_time) {
    
    int N = matrix[0].size();

    // flatten matrix for GPU processing
    DArray flatMatrix(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            flatMatrix[i * N + j] = matrix[i][j];
        }
    }

    // allocate device memory
    double *d_matrix, *d_vector, *d_result;
    cudaMalloc(&d_matrix, N * N * sizeof(double));
    cudaMalloc(&d_vector, N * sizeof(double));
    cudaMalloc(&d_result, N * sizeof(double));

    // copy data to device
    cudaMemcpy(d_matrix, flatMatrix.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // launch the kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    matrixVectorGPU<<<numBlocks, blockSize>>>(d_matrix, d_vector, d_result, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGPU = end - start;
    GPU_time = durationGPU.count();

    // copy the result back to host
    cudaMemcpy(resultGPU.data(), d_result, N * sizeof(double), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
}



int main(int argc, char** argv) {
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix/vector size (N)>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);

    printf("N = %d\n", N);
    TwoDDArray matrix = generate_normal_matrix(N, N);
    DArray input = gen_wave(N);

    DArray result_GPU(N);
    DArray par_result_CPU(N);
    DArray serial_result(N);

    int num_threads = std::thread::hardware_concurrency();

    int nb_tries = 30;
    printf("Over %d tries:\n", nb_tries);

    //------------------- Serial CPU --------------------
    auto start = std::chrono::high_resolution_clock::now();
    serial_result = serial_matrix_vect_mul(matrix, input);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> dur_serial_cpu = end - start;

    //------------------- Parallel CPU --------------------
    start = std::chrono::high_resolution_clock::now();
    par_result_CPU = parallel_matrix_mul(matrix, input, num_threads);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> dur_par_cpu = end - start;

    //----------------------- GPU ------------------------
    std::chrono::duration<float> dur_gpu = std::chrono::duration<float>::zero();
    double inner_gpu_time = 0;
    for (int tries = 0; tries < nb_tries; ++tries) {
        start = std::chrono::high_resolution_clock::now();
        matrix_vect_mul_GPU(matrix, input, result_GPU, inner_gpu_time);
        end = std::chrono::high_resolution_clock::now();
        dur_gpu = end - start;
    }


    std::cout << "Single-threaded CPU time: " << dur_serial_cpu.count() << " seconds" << std::endl;
    std::cout << "Multithreaded CPU time: " << dur_par_cpu.count() << " seconds" << std::endl;
    std::cout << "GPU time (load included): " << dur_gpu.count() << " seconds" << std::endl;
    std::cout << "GPU time (load excluded): " << inner_gpu_time << " seconds" << std::endl;

    std::cout << "Single-threaded CPU / GPU ratio (load included): " << dur_serial_cpu.count() / dur_gpu.count() << std::endl;
    std::cout << "Single-threaded CPU / GPU ratio (load excluded): " << dur_serial_cpu.count() / inner_gpu_time  << std::endl;
    //std::cout << "Multithreaded CPU / GPU ratio: " << dur_par_cpu.count() / dur_gpu.count() << std::endl;

    return 0;
}