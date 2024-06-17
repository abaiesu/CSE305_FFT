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


void flip(IArray& indices){
    std::reverse(indices.begin(), indices.end());
}


int get_Lj(int j, IArray &dimensions){
    
    if(j == 0){
        return 1;
    }
    IArray flipped = dimensions;
    flip(flipped);
    int Lj = 1;
    for (int i = 0; i < j; ++i){
        Lj *= flipped[i];
    }
    return Lj;

}


int get_1D_index(IArray &coords, IArray &dimensions) {
    int index = 0;
    for (int i = 0; i < dimensions.size(); ++i) {
        int partial = 1;
        for(int j = 0; j < i; ++j) {
            partial *= dimensions[j];
        }
        index += partial * coords[i];
    }
    return index;
}


IArray get_ND_index(int index, IArray& dimensions){
    int p = dimensions.size(); 
    IArray indices(p);
    for (int i = 0; i < p; i++){
        indices[i] = index % dimensions[i];
        index = index / dimensions[i];
    }
    return indices;
}




__global__ void gfft_kernel(Complex *x, int N, double alpha) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        if (N <= 1) {
            return;
        }

        if (N == 2) {
            cuDoubleComplex even = x[0];
            cuDoubleComplex odd = x[1];
            cuDoubleComplex Twiddle = make_cuDoubleComplex(cos(-2 * PI * alpha / N), sin(-2 * PI * alpha / N));
            cuDoubleComplex t = cMul(Twiddle, odd);
            x[0] = cAdd(even, t);
            x[1] = cCSub(even, t);
            return;
        }

        // Divide
        int halfN = N / 2;
        cuDoubleComplex *even = new cuDoubleComplex[halfN];
        cuDoubleComplex *odd = new cuDoubleComplex [halfN];
        for (int i = 0; i < halfN; ++i) {
            even[i] = x[i * 2];
            odd[i] = x[i * 2 + 1];
        }

        // Conquer
        gfft_kernel<<<1, 1>>>(even, halfN, alpha);
        gfft_kernel<<<1, 1>>>(odd, halfN, alpha);
        cudaDeviceSynchronize();

        // Combine
        cuDoubleComplex  W_N = {cosf(-2 * PI / N), sinf(-2 * PI / N)};
        cuDoubleComplex  W = {1.0, 0.0};
        cuDoubleComplex  W_alpha = {cosf(-2 * PI * alpha / N), sinf(-2 * PI * alpha / N)};
        for (int k = 0; k < halfN; ++k) {
            cuDoubleComplex inter = cMul(W, W_alpha);
            cuDoubleComplex t = cMul(inter, odd[k]);
            x[k] = cAdd(even[k], t);
            x[k + halfN] = cCSub(even[k], t);
            W = cMul(W, W_N);
        }

        delete[] even;
        delete[] odd;
    }
}



void index_reversal_worker(CArray &input, CArray &res, int start, int end, IArray &dimensions) {
    
    for(int i = start; i < end; i++){
        // Get the N-dimensional index (i.e. the coordinates)
        IArray indices = get_ND_index(i, dimensions);
        flip(indices);
        int index = get_1D_index(indices, dimensions);
        res[index] = input[i];
    }
}



__global__ void block_worker_kernel(cuDoubleComplex* input, cuDoubleComplex* res, int starting_block_index, int stage,
                                   int num_blocks_to_process, int block_size, int stride, int* dimensions, int* L_js) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_blocks_to_process) {
        int block_index = starting_block_index + tid;
        int start = block_index * block_size;

        cuDoubleComplex* input_temp = input + start;
        IArray dims(5); // Using std::array
        for (int i = 0; i < 5; ++i) {
            dims[i] = dimensions[i];
        }
        IArray temp = get_ND_index(start, dims);
        int* coords = temp.data();

        int g_jm1 = 0;
        for (int l = 1; l <= stage - 1; ++l) {
            int k_index = blockDim.x - stage + l;  // Assuming blockDim.x is the size of dimensions array
            int L_term = L_js[l - 1];
            int K_term = coords[k_index];
            g_jm1 += K_term * L_term;
        }

        double L = L_js[stage - 1];
        double alpha = (stage == 0) ? 0 : g_jm1 / L;

        // Perform FFT directly on GPU
        gfft_kernel<<<1, 1>>>(input_temp, alpha, dimensions[0]);
        
        // Store the result back to res array with proper stride
        for (int i = 0; i < block_size; ++i) {
            res[block_index + i * stride] = input_temp[i];
        }
    }
}

void parallel_dft_GPU(CArray& input, IArray& dimensions, int num_threads) {
    ull N = input.size();
    ull sum_dims = get_Lj(dimensions.size(), dimensions);

    if (N != sum_dims) {
        std::cerr << "The size of the input array is not equal to the product of the dimensions" << std::endl;
        printf("N = %d, prod_dim = %d\n", N, sum_dims);
        return;
    }

    CArray inter_res(N); //store the intermediae result
    int p = dimensions.size();

    int working_threads = (num_threads > N) ? N : num_threads;
    std::vector<std::thread> threads (working_threads);
    int num_clients = N / working_threads;
    for (int i = 0; i < working_threads; ++i){
        int start = i * num_clients;
        int end = (i == working_threads - 1) ? N : (i + 1) * num_clients;
        threads[i] = std::thread(index_reversal_worker, std::ref(input), std::ref(inter_res), start, end, std::ref(dimensions));
    }

    for (int i = 0; i < working_threads; ++i) {
        threads[i].join();
    }

    input = inter_res;

    int threadsPerBlock;

    // Allocate memory on GPU
    cuDoubleComplex* d_input;
    cudaMalloc((void**)&d_input, N * sizeof(cuDoubleComplex));
    cudaMemcpy(d_input, input.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Allocate memory for intermediate results on GPU
    cuDoubleComplex* d_intermediate_res;
    cudaMalloc((void**)&d_intermediate_res, N * sizeof(cuDoubleComplex));

    // Precompute L_js on CPU
    IArray L_js(dimensions.size() + 1);
    L_js[0] = 1;
    for (int j = 1; j <= dimensions.size(); ++j) {
        L_js[j] = pow(dimensions[0], j);
    }

    // Launch block worker kernel for each stage
    int num_blocks = N / dimensions[0];
    int blocksPerGrid_block_worker = (num_blocks + threadsPerBlock - 1) / threadsPerBlock;
    for (int stage = 1; stage <= dimensions.size(); ++stage) {
        block_worker_kernel<<<blocksPerGrid_block_worker, threadsPerBlock>>>(d_intermediate_res, d_input, 0, stage,
                                                                              num_blocks, dimensions[0], num_blocks,
                                                                              dimensions.data(), L_js.data());
        cudaDeviceSynchronize();
    }

    // Copy results back to CPU
    cudaMemcpy(input.data(), d_input, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_intermediate_res);
}



void parallel_dft(CArray &input, IArray &dimensions, int num_threads) {
    
    ull N = input.size();
    ull sum_dims = get_Lj(dimensions.size(), dimensions);

    if (N != sum_dims){
        std::cerr << "The size of the input array is not equal to the product of the dimensions" << std::endl;
        printf("N = %d, prod_dim = %d\n", N, sum_dims);
        return;
    }


    CArray inter_res(N); //store the intermediae result
    int p = dimensions.size();

    // 1. Index-reversal permutation (in parallel)

    int working_threads = (num_threads > N) ? N : num_threads;
    std::vector<std::thread> threads (working_threads);
    int num_clients = N / working_threads;
    for (int i = 0; i < working_threads; ++i){
        int start = i * num_clients;
        int end = (i == working_threads - 1) ? N : (i + 1) * num_clients;
        threads[i] = std::thread(index_reversal_worker, std::ref(input), std::ref(inter_res), start, end, std::ref(dimensions));
    }

    for (int i = 0; i < working_threads; ++i) {
        threads[i].join();
    }

    input = inter_res;

    // Precompute the L_js
    IArray L_js(p + 1);
    L_js[0] = 1;
    for (int j = 1; j <= p; ++j){
        L_js[j] = pow(dimensions[0], j); 
    }

    // 2. For each dimension p
    int num_blocks = N / dimensions[0]; // number of blocks
    working_threads = (num_threads > num_blocks) ? num_blocks : num_threads;
    int num_blocks_per_thread = num_blocks / working_threads;
    int block_size = dimensions[0]; // size of each block
    for (int stage = 1; stage <= p; ++stage) {

        /* after the first stage, we shift the dimensions to the left
        ex :    input dimension : 2 x 4 x 2
                stage 1 = 2 x 4 x 2
                stage 2 = 4 x 2 x 2
                stage 3 = 2 x 2 x 4
        */
        
        /*if (stage != 1){ 
            shiftLeft(dimensions);
        }
        
        ONLY RELEVANT WHEN THE Ns ARENT ALL THE SAME
        */
    
        CArray inter_res(N); //get a new array to store the intermediate vector
        std::vector<std::thread> threads (working_threads); //one thread per block

        for (int thread_index = 0; thread_index < working_threads; ++thread_index) { // for each block (in parallel)
            
            int num_blocks_to_process = (thread_index == working_threads - 1) ? num_blocks_per_thread + num_blocks % working_threads : num_blocks_per_thread;
            int starting_block_index = thread_index * num_blocks_per_thread;
            threads[thread_index] = std::thread(block_worker, std::ref(input), std::ref(inter_res), 
                                                        starting_block_index, stage,
                                                        num_blocks_to_process, block_size, 
                                                        num_blocks, std::ref(dimensions),
                                                        std::ref(L_js));

        
        }
        
        for (int i = 0; i < working_threads; ++i) {
            threads[i].join();
        }


        input = inter_res;
    }

}


int main() {
    
    int num_threads = 20;

    // Input data
    
    int M = 16;
    int p = 5;
    ull N = pow(M, p);
    IArray dimensions(p, M);
    CArray input = gen_temp(N);

    auto start = std::chrono::high_resolution_clock::now();
    parallel_dft_GPU(input, dimensions, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFastDCT = end - start;
    std::cout << "GPU time: " << durationFastDCT.count() << " seconds\n";

    /*start = std::chrono::high_resolution_clock::now();
    DArray outGPU = GPU_parallel_dct(input, num_threads);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationParallelDCT = end - start;
    std::cout << "GPU time: " << durationParallelDCT.count() << " seconds\n";

    bool are_equal = are_arrays_equal(outCPU, outGPU);
    std::cout << "Correct ? " << (are_equal ? "Yes" : "No") << std::endl;*/

    return 0;
}
