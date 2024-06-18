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
#include "dft.h"

__device__ const double PI_gpu = 3.14159265358979323846;

__device__ struct OwnComplex {
    float real, imag;
};

__device__ typedef unsigned long long int ull;


__device__
int* get_ND_index_device(int index, int M, int p){ 
    int* indices = new int[p];
    for (int i = 0; i < p; i++){
        indices[i] = index % M;
        index = index / M;
    }
    return indices;
}


// CUDA device function declaration
__device__
void gfft_device(OwnComplex* x, double alpha, size_t N) {
    
    if (N <= 1) {
        return;
    }

    // Divide
    OwnComplex* even = new OwnComplex[N / 2];
    OwnComplex* odd = new OwnComplex[N / 2];
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Conquer
    gfft_device(even, alpha, N / 2);
    gfft_device(odd, alpha, N / 2);

    // Combine
    OwnComplex W_N;
    W_N.real = cos(-2 * PI_gpu / N);
    W_N.imag = sin(-2 * PI_gpu / N);
    OwnComplex W;
    W.real = 1.0;
    W.imag = 0.0;
    OwnComplex W_alpha;
    W_alpha.real = cos(-2 * PI_gpu * alpha / N);
    W_alpha.imag = sin(-2 * PI_gpu * alpha / N);
    for (size_t k = 0; k < N / 2; ++k) {
        OwnComplex t;
        t.real = W.real * W_alpha.real - W.imag * W_alpha.imag;
        t.imag = W.real * W_alpha.imag + W.imag * W_alpha.real;
        x[k].real = even[k].real + t.real * odd[k].real - t.imag * odd[k].imag;
        x[k].imag = even[k].imag + t.real * odd[k].imag + t.imag * odd[k].real;
        x[k + N / 2].real = even[k].real - t.real * odd[k].real + t.imag * odd[k].imag;
        x[k + N / 2].imag = even[k].imag - t.real * odd[k].imag - t.imag * odd[k].real;
        W.real = W.real * W_N.real - W.imag * W_N.imag;
        W.imag = W.real * W_N.imag + W.imag * W_N.real;
    }

    delete[] even;
    delete[] odd;
}



__global__ 
void block_worker(OwnComplex* input, OwnComplex* res, int stage, 
                  int M, int p, int* L_js) {

    int thread_index = threadIdx.x + blockIdx.x * blockDim.x; //thread index = block_index

    int N = pow(M, p);

    int stride = pow(M, p-1);

    OwnComplex* input_temp = new OwnComplex[M]; //must be freed

    size_t start = thread_index*M; //absolute index of the head of the block within the input
    // Load the parts of the input we want to treat
    for (int i = 0; i < M; ++i) {
        input_temp[i] = input[start + i];
    }
    
    
    int* coords = get_ND_index_device(start, M, p); //must be freed

    int g_jm1 = 0;
    for (int l = 1; l <= stage - 1; ++l){
        int k_index = p - stage + l;
        int L_term = L_js[l-1];
        int K_term = coords[k_index];
        g_jm1 += K_term * L_term; 
    }

    
    double L  = L_js[stage - 1];
    double alpha = (stage == 0) ? 0 : g_jm1 / L;

    gfft_device(input_temp, alpha, N);
    
    //store the result
    for(int i = 0; i < M; i++){
        res[thread_index + i*stride] = input_temp[i];
    }

    delete[] coords;
    delete[] input_temp;

}







void parallel_dft_GPU(std::vector<OwnComplex> &input, int M, int p) {
    
    ull N = input.size();
    ull prod = pow(M, p);

    if (N != prod){
        std::cerr << "The size of the input array is not equal to the product of the dimensions" << std::endl;
        printf("N = %d, prod = %d\n", N, prod);
        return;
    }

    std::vector<OwnComplex> inter_res(N); //store the intermediae result

    // 1. Index-reversal permutation (in parallel)

    IArray dimensions (p, M);

    for(int i = 0; i < N; i++){
        // Get the N-dimensional index (i.e. the coordinates)
        IArray indices = get_ND_index(i, dimensions);
        flip(indices);
        int index = get_1D_index(indices, dimensions);
        inter_res[index] = input[i];
    }

    //--------------------------------------------------------------------------

    // 2. Precompute the L_js
    int* L_js = new int[p + 1];
    L_js[0] = 1;
    for (int j = 1; j <= p; ++j){
        L_js[j] = pow(M, j); 
    }

    //--------------------------------------------------------------------------


    // 3. For each dimension p


    OwnComplex *d_input, *d_res;
    int* d_L_js;
    cudaMalloc(&d_input, N * sizeof(OwnComplex));
    cudaMalloc(&d_res, N * sizeof(OwnComplex));
    cudaMalloc(&d_L_js, (p+1) * sizeof(int));

    cudaMemcpy(d_input, inter_res.data(), N * sizeof(OwnComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L_js, L_js, (p+1) * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int needed_nb_threads = pow(M, p-1);
    int numBlocks = (needed_nb_threads + blockSize - 1) / blockSize;

    for (int stage = 1; stage <= p; stage++) {
        block_worker<<<numBlocks, blockSize>>>(d_input, d_res, stage, M, p, d_L_js);
        cudaDeviceSynchronize();
        std::swap(d_input, d_res);  // Swap the input and result pointers
    }

    cudaMemcpy(input.data(), d_input, N * sizeof(OwnComplex), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_res);
    cudaFree(d_L_js);
    delete[] L_js;
}




CArray convertToStdComplex(const std::vector<OwnComplex>& input) {
    CArray output;
    output.reserve(input.size()); // Reserve space to avoid multiple allocations
    for (const auto& ownComplex : input) {
        output.emplace_back(static_cast<double>(ownComplex.real), static_cast<double>(ownComplex.imag));
    }
    return output;
}



int main(int argc, char* argv[]) {
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <M> <p>" << std::endl;
        return 1;
    }

    // Parse command line arguments for M and p
    int M = std::atoi(argv[1]);
    int p = std::atoi(argv[2]);
    IArray dims(p, M);
    
    // Example input data (a cube of MxMxM elements, flattened into a 1D array)
    int N = pow(M, p); // Total number of elements
    std::vector<OwnComplex> input(N);
    for (int i = 0; i < N; ++i) {
        input[i].real = i;   // Example initialization of real part
        input[i].imag = 0.0; // Example initialization of imaginary part
    }

    CArray input_1 = convertToStdComplex(input);

    // Timing variables
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double> elapsed_time;

    // Start timing
    start_time = std::chrono::high_resolution_clock::now();

    // Call parallel_dft function
    parallel_dft_GPU(input, M, p);

    // End timing
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

    std::cout << "GPU DFT took " << elapsed_time.count() << " seconds." << std::endl;



    //---------------------------------------------------------------------------------------------

    start_time = std::chrono::high_resolution_clock::now();

    int num_threads = 20;
    // Call parallel_dft function
    parallel_dft(input_1, dims, num_threads);
    // End timing
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

    std::cout <<"CPU took " << elapsed_time.count() << " seconds." << std::endl;

    CArray gpu_res = convertToStdComplex(input);
    
    /*if (are_arrays_equal<CArray>(gpu_res, input_1)) {
        std::cout << "Correct output :)" << std::endl;
    } else {
        std::cout << "Incorrect output :(" << std::endl;
    }*/

    
    return 0;
}