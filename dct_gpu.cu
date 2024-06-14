#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include </usr/local/cuda/include/cuda.h>
#include "helpers.h"

typedef std::complex<double> Complex;
typedef std::vector<Complex> CArray;
typedef std::vector<int> IArray;
typedef std::vector<double> DArray;
typedef unsigned long long int ull;
typedef std::vector<CArray> TwoDCArray;
typedef std::vector<DArray> TwoDDArray;

__device__ const double PI_gpu = 3.14159265358979323846;

//-------------------------- DCT-II based on DFT PARALLEL -----------------------------

__global__ void worker_dct_kernel(cuDoubleComplex* V, double* result, int N, int start, int end) {
    int k = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (k < end) {
        cuDoubleComplex factor = make_cuDoubleComplex(cos(-PI_gpu * k / (2.0 * N)) * 2.0, sin(-PI_gpu * k / (2.0 * N)) * 2.0);
        if (N % 2 != 0 && k == 0) {
            factor = make_cuDoubleComplex(cuCreal(factor) / sqrt((float) 2), cuCimag(factor) / sqrt((float) 2));
        }
        cuDoubleComplex Vk = V[k];
        cuDoubleComplex result_cu = cuCmul(Vk, factor);
        result[k] = cuCreal(result_cu) * sqrt((float) (1.0 / (2.0 * N)));
    }
}



DArray GPU_parallel_dct(const DArray& input, int num_threads) {
    int N = input.size();
    CArray v(N);

    for (int i = 0; i <= (N - 1) / 2; ++i) {
        v[i] = input[2 * i];
        v[N / 2 + i] = input[N - 1 - 2 * i];
    }

    // Allocate device memory
    cuDoubleComplex* d_V;
    double* d_result;
    cudaMalloc(&d_V, N * sizeof(cuDoubleComplex));
    cudaMalloc(&d_result, N * sizeof(double));

    // Copy data to device
    std::vector<cuDoubleComplex> v_cu(N);
    for (int i = 0; i < N; ++i) {
        //v_cu[i] = to_cuComplex(v[i]);
        v_cu[i] = make_cuDoubleComplex(v[i].real(), v[i].imag());
    }
    cudaMemcpy(d_V, v_cu.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Launch the worker_dct_kernel on the GPU
    int num_blocks = (N + num_threads - 1) / num_threads;
    worker_dct_kernel<<<num_blocks, num_threads>>>(d_V, d_result, N, 0, N);

    // Copy result back to host
    DArray result(N);
    cudaMemcpy(result.data(), d_result, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_V);
    cudaFree(d_result);

    // Scale the output
    result[0] *= double(1 / std::sqrt(2));

    return result;
}


void worker_dct(CArray& V, DArray& result, int start, int end) {
    int N = result.size();
    for (int k = start; k < end; ++k) {
        Complex factor = std::exp(Complex(0, -PI * k / (2.0 * N))) * 2.0;
        if (N % 2 != 0 && k == 0) {
            factor /= std::sqrt(2);
        }
        result[k] = (V[k] * factor).real() * std::sqrt(1.0 / (2.0 * N));
    }
}

DArray parallel_dct(const DArray& input, IArray dimensions, int num_threads) {
    int N = input.size();
    CArray v(N);

    for (int i = 0; i <= (N - 1) / 2; ++i) {
        v[i] = input[2 * i];
        v[N / 2 + i] = input[N - 1 - 2 * i];
    }

    // Assuming parallel_dft is defined somewhere
    // parallel_dft(v, dimensions, num_threads);

    // Compute the DCT-II result
    DArray result(N);
    std::vector<std::thread> threads(num_threads);
    int num_blocks = N / num_threads;
    int start, end;
    for (int i = 0; i < num_threads; ++i) {
        start = i * num_blocks;
        end = (i == num_threads - 1) ? N : (i + 1) * num_blocks;
        threads[i] = std::thread(worker_dct, std::ref(v), std::ref(result), start, end);
    }

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    // Scale the output
    result[0] *= double(1 / std::sqrt(2));

    return result;
}

DArray gen_wave(ull n) {
    DArray signal(n);
    double pi = 3.14159265358979323846;

    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / (n - 1);
        signal[i] = std::cos(2 * 97 * pi * t) + std::cos(2 * 777 * pi * t);
    }

    return signal;
}

int main() {
    
    int num_threads = 20;

    // Input data
    for(int k = 0; k < 10; k++){
        int M = 16;
        int p = 6;
        ull N = pow(M, p);
        IArray dimensions(p, M);
        DArray input = gen_wave(N);

        auto start = std::chrono::high_resolution_clock::now();
        DArray outCPU = parallel_dct(input, dimensions, num_threads);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationFastDCT = end - start;
        std::cout << "CPU time: " << durationFastDCT.count() << " seconds\n";

        start = std::chrono::high_resolution_clock::now();
        DArray outGPU = GPU_parallel_dct(input, num_threads);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationParallelDCT = end - start;
        std::cout << "GPU time: " << durationParallelDCT.count() << " seconds\n";

        bool are_equal = are_arrays_equal(outCPU, outGPU);
        std::cout << "Correct ? " << (are_equal ? "Yes" : "No") << std::endl;
    }

    return 0;
}