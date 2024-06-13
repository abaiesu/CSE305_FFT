#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <thread>
#include "dft.h"
#include "helpers.h"

using CArray = std::vector<std::complex<double>>;
using IArray = std::vector<int>;

// Naive DCT-II implementation
std::vector<double> naiveDCT(const std::vector<double>& input) {
    int N = input.size();
    std::vector<double> output(N);

    // Compute the DCT-II using the naive approach
    for (int k = 0; k < N; ++k) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) {
            sum += input[n] * std::cos(PI * k * (2.0 * n + 1) / (2.0 * N));
        }
        output[k] = 2.0 * sum;
    }

    // Scale the output
    for (int k = 0; k < N; ++k) {
        double f = (k == 0) ? std::sqrt(1.0 / (4.0 * N)) : std::sqrt(1.0 / (2.0 * N));
        output[k] *= f;
    }

    return output;
}

// Naive IDCT-II implementation
std::vector<double> naiveIDCT(const std::vector<double>& input) {
    int N = input.size();
    std::vector<double> output(N);

    // Compute the IDCT-II using the naive approach
    for (int n = 0; n < N; ++n) {
        double sum = input[0] * std::sqrt(1.0 / (4.0 * N));
        for (int k = 1; k < N; ++k) {
            sum += input[k] * std::cos(PI * k * (2.0 * n + 1) / (2.0 * N)) * std::sqrt(1.0 / (2.0 * N));
        }
        output[n] = 2.0 * sum;
    }

    return output;
}

// Fast DCT-II implementation
std::vector<double> dctFast(const std::vector<double>& input) {
    int N = input.size();
    std::vector<double> v(N, 0.0);

    // Prepare the input for FFT
    for (int i = 0; i <= (N - 1) / 2; ++i) {
        v[i] = input[2 * i];
    }

    if (N % 2) {  // odd length
        for (int i = 0; i < (N - 1) / 2; ++i) {
            v[(N - 1) / 2 + 1 + i] = input[N - 2 - 2 * i];
        }
    } else {  // even length
        for (int i = 0; i < N / 2; ++i) {
            v[N / 2 + i] = input[N - 1 - 2 * i];
        }
    }

    // Perform FFT
    CArray V(v.begin(), v.end());
    serial_dft(V);

    // Compute the DCT-II result
    std::vector<double> result(N);
    for (int k = 0; k < N; ++k) {
        std::complex<double> factor = std::exp(std::complex<double>(0, -PI * k / (2.0 * N))) * 2.0;
        if (N % 2 != 0 && k == 0) {
            factor /= std::sqrt(2);
        }
        result[k] = (V[k] * factor).real();
    }

    // Scale the output
    for (int k = 0; k < N; ++k) {
        if (k == 0) {
            result[k] *= std::sqrt(1.0 / (4.0 * N));
        } else {
            result[k] *= std::sqrt(1.0 / (2.0 * N));
        }
    }

    return result;
}

// Fast IDCT-II implementation
std::vector<double> idctFast(const std::vector<double>& dctInput) {
    int N = dctInput.size();
    std::vector<std::complex<double>> shiftGrid(N);

    // Prepare the shift grid for IFFT
    for (int i = 0; i < N; ++i) {
        shiftGrid[i] = std::sqrt(2.0 * N) * std::exp(std::complex<double>(0, PI * i / (2.0 * N)));
    }
    shiftGrid[0] /= std::sqrt(2.0);

    // Apply the shift grid to the input
    CArray vTmpComplex(N);
    for (int i = 0; i < N; ++i) {
        vTmpComplex[i] = shiftGrid[i] * dctInput[i];
    }

    // Perform IFFT
    IArray dimensions;
    int num_threads = 1;
    idft(vTmpComplex, dimensions, num_threads);

    // Reconstruct the original signal
    std::vector<double> x(N);
    for (int i = 0; i < N / 2; ++i) {
        x[2 * i] = vTmpComplex[i].real();
        x[2 * i + 1] = vTmpComplex[N - i - 1].real();
    }

    return x;
}

// Parallelized fast DCT-II implementation
void dctParallelWorker(const std::vector<double>& input, std::vector<double>& v, int start, int end) {
    int N = input.size();
    for (int i = start; i < end; ++i) {
        if (i <= (N - 1) / 2) {
            v[i] = input[2 * i];
        } else if (N % 2) {  // odd length
            v[i] = input[N - 2 - 2 * (i - (N - 1) / 2 - 1)];
        } else {  // even length
            v[i] = input[N - 1 - 2 * (i - N / 2)];
        }
    }
}

std::vector<double> dctParallel(const std::vector<double>& input) {
    int N = input.size();
    std::vector<double> v(N, 0.0);

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    int chunk_size = N / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? N : (t + 1) * chunk_size;
        threads.push_back(std::thread(dctParallelWorker, std::cref(input), std::ref(v), start, end));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Perform FFT
    CArray V(v.begin(), v.end());
    serial_dft(V);

    // Compute the DCT-II result
    std::vector<double> result(N);
    for (int k = 0; k < N; ++k) {
        std::complex<double> factor = std::exp(std::complex<double>(0, -PI * k / (2.0 * N))) * 2.0;
        if (N % 2 != 0 && k == 0) {
            factor /= std::sqrt(2);
        }
        result[k] = (V[k] * factor).real();
    }

    // Scale the output
    for (int k = 0; k < N; ++k) {
        if (k == 0) {
            result[k] *= std::sqrt(1.0 / (4.0 * N));
        } else {
            result[k] *= std::sqrt(1.0 / (2.0 * N));
        }
    }

    return result;
}

// Parallelized fast IDCT-II implementation
void idctParallelWorker(const std::vector<std::complex<double>>& vTmpComplex, std::vector<double>& x, int start, int end) {
    int N = vTmpComplex.size();
    for (int i = start; i < end; ++i) {
        if (i < N / 2) {
            x[2 * i] = vTmpComplex[i].real();
            x[2 * i + 1] = vTmpComplex[N - i - 1].real();
        }
    }
}

std::vector<double> idctParallel(const std::vector<double>& dctInput) {
    int N = dctInput.size();
    std::vector<std::complex<double>> shiftGrid(N);

    // Prepare the shift grid for IFFT
    for (int i = 0; i < N; ++i) {
        shiftGrid[i] = std::sqrt(2.0 * N) * std::exp(std::complex<double>(0, PI * i / (2.0 * N)));
    }
    shiftGrid[0] /= std::sqrt(2.0);

    // Apply the shift grid to the input
    CArray vTmpComplex(N);
    for (int i = 0; i < N; ++i) {
        vTmpComplex[i] = shiftGrid[i] * dctInput[i];
    }

    // Perform IFFT
    IArray dimensions;
    int num_threads = 1;
    idft(vTmpComplex, dimensions, num_threads);

    // Reconstruct the original signal
    std::vector<double> x(N, 0.0);
    
    num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = N / (2 * num_threads);
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? N / 2 : (t + 1) * chunk_size;
        threads.push_back(std::thread(idctParallelWorker, std::cref(vTmpComplex), std::ref(x), start, end));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return x;
}

// Helper function to convert std::vector<double> to CArray
CArray vectorToCArray(const std::vector<double>& vec) {
    CArray result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = std::complex<double>(vec[i], 0.0);
    }
    return result;
}

int main() {
    // Input data
    std::vector<double> input = {1.0, 2.0, 3.0, 4.0};

    // Perform naive DCT and IDCT
    std::vector<double> dctNaiveOutput = naiveDCT(input);
    std::cout << "Naive DCT output:\n";
    for (double val : dctNaiveOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::vector<double> idctNaiveOutput = naiveIDCT(dctNaiveOutput);
    std::cout << "Naive IDCT output:\n";
    for (double val : idctNaiveOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Perform fast DCT and IDCT
    std::vector<double> dctFastOutput = dctFast(input);
    std::cout << "Fast DCT output:\n";
    for (double val : dctFastOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::vector<double> idctFastOutput = idctFast(dctFastOutput);
    std::cout << "Fast IDCT output:\n";
    for (double val : idctFastOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Perform parallelized fast DCT
    std::vector<double> dctParallelOutput = dctParallel(input);
    std::cout << "Parallelized Fast DCT output:\n";
    for (double val : dctParallelOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Perform parallelized fast IDCT
    std::vector<double> idctParallelOutput = idctParallel(dctParallelOutput);
    std::cout << "Parallelized Fast IDCT output:\n";
    for (double val : idctParallelOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Convert vectors to CArray for comparison
    CArray inputCArray = vectorToCArray(input);
    CArray idctNaiveCArray = vectorToCArray(idctNaiveOutput);
    CArray idctFastCArray = vectorToCArray(idctFastOutput);
    CArray idctParallelCArray = vectorToCArray(idctParallelOutput);

    // Compare results with original input
    bool are_equal_naive = are_arrays_equal(inputCArray, idctNaiveCArray);
    bool are_equal_fast = are_arrays_equal(inputCArray, idctFastCArray);
    bool are_equal_parallel = are_arrays_equal(inputCArray, idctParallelCArray);

    std::cout << "Naive IDCT result matches original input: " << (are_equal_naive ? "Yes" : "No") << std::endl;
    std::cout << "Fast IDCT result matches original input: " << (are_equal_fast ? "Yes" : "No") << std::endl;
    std::cout << "Parallel IDCT result matches original input: " << (are_equal_parallel ? "Yes" : "No") << std::endl;

    return 0;
}

