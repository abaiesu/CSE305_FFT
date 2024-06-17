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
#include "dct.h"

__device__ const double PI_gpu = 3.14159265358979323846;

//-------------------------- DCT-II based on DFT PARALLEL -----------------------------

__global__ void worker_dct_kernel(cuDoubleComplex* V, double* result, int N) {
    
    int k = blockIdx.x * blockDim.x + threadIdx.x ;//+ start;
    if (k < N) {
        cuDoubleComplex factor = make_cuDoubleComplex(cos(-PI_gpu * k / (2.0 * N)) * 2.0, sin(-PI_gpu * k / (2.0 * N)) * 2.0);
        if (N % 2 != 0 && k == 0) {
            factor = make_cuDoubleComplex(cuCreal(factor) / sqrt((float) 2), cuCimag(factor) / sqrt((float) 2));
        }
        cuDoubleComplex Vk = V[k];
        cuDoubleComplex result_cu = cuCmul(Vk, factor);
        result[k] = cuCreal(result_cu) * sqrt((float) (1.0 / (2.0 * N)));
    }

}



DArray GPU_parallel_dct(const DArray& input, int M, int p, int num_threads) {
    
    int N = input.size();
    CArray v(N);

    for (int i = 0; i <= (N - 1) / 2; ++i) {
        v[i] = input[2 * i];
        v[N / 2 + i] = input[N - 1 - 2 * i];
    }

    IArray dims (p, M);
    parallel_dft(v, dims, num_threads);
    
    
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

    const size_t THREADS_PER_BLOCK = 256;
    size_t BLOCKS_NUM = N / THREADS_PER_BLOCK;
    worker_dct_kernel<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(d_V, d_result, N);

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




int main(int argc, char** argv) {
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <M> <p>" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int p = std::atoi(argv[2]);

    
    int num_threads = 20;
    ull N = pow(M, p);
    printf("N = %lld\n", N);
    IArray dimensions(p, M);
    DArray input = gen_wave(N);
    DArray out1, out2;

    auto start = std::chrono::high_resolution_clock::now();
    out1 = parallel_dct(input, dimensions, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur_cpu = end - start;
    std::cout << "Multithreaded CPU time: " << dur_cpu.count() << " seconds\n";

    // Input data
    int nb_tries = 30;
    std::chrono::duration<double> dur_gpu;
    for(int k = 0; k < nb_tries; k++){
        start = std::chrono::high_resolution_clock::now();
        out2 = GPU_parallel_dct(input, M, p, num_threads);
        end = std::chrono::high_resolution_clock::now();
        dur_gpu = end - start;
    }

    printf("After %d rounds of GPU warm-up:\n", nb_tries);
    std::cout << "GPU time: " << dur_gpu.count() << " seconds\n";
    std::cout << "CPU time / GPU time: " <<  dur_cpu.count()/dur_gpu.count() << " seconds\n";


    if (!are_arrays_equal(out1, out2)){
        printf("NOT THE SAME\n");
    }
   
    return 0;
}