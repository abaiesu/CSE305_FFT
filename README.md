# CSE305_FFT
CSE305 final project : parallel FFT

## 1. `dft.cpp`

- **Description:** Serial and parallel 1D-DFT, 1D-IDFT, 2D-DFT, 2D-IDFT

## 2. `dct.cpp`

- **Description:** Naive, FastSerial and FastParallel DCT, ICFT

## 3. `CS.cpp`

- **Description:** Serial and parrallel OMP

## 4. `CS_demo.cpp`

- **Description:** Implementation of a CS procedure for a generic signal (undersampling + recovery) + classic DFT sparsification

## 5. `perf_test.cpp`

- **Description:** Various speed and correctness tests, view usage with ./perf_test

## 6. `matmul_gpu.cu`

- **Description:** CUDA implementation of matrix vector multiplication + speed tests

## 7. `dct_gpu.cu`

- **Description:** CUDA implementation a partial parallelization of the DCT on the GPU + speed tests

## 8. `dft_gpu.cu`

- **Description:** Attempt to parallel DFT implemented in CUDA


## 9. `helpers.cpp`

- **Description:** Periodic signal generation, files reading and writing, random matrix generation, serial and parallel matrix-vector multiplication, equality checks between 2 matrices, equality checks between 2 vectors equal, sparsification 
