#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include "dft.h"
#include "helpers.h"

using CArray = std::vector<std::complex<double>>;
using IArray = std::vector<int>;

// Naive DCT-II implementation
DArray naiveDCT(const DArray& input) {
    int N = input.size();
    DArray output(N);

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
DArray naiveIDCT(const DArray& input) {
    int N = input.size();
    DArray output(N);

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
DArray dctFast(const DArray& input) {
    int N = input.size();
    DArray v(N, 0.0);

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
DArray idctFast(const DArray& dctInput) {
    int N = dctInput.size();
    CArray shiftGrid(N);

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
    DArray x(N);
    for (int i = 0; i < N / 2; ++i) {
        x[2 * i] = vTmpComplex[i].real();
        x[2 * i + 1] = vTmpComplex[N - i - 1].real();
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

/*int main() {
    // Input data
    int N = pow(2, 12);
    DArray input = gen_wave(N);

    // Perform naive DCT and IDCT
    DArray dctNaiveOutput = naiveDCT(input);
    /*std::cout << "Naive DCT output:\n";
    for (double val : dctNaiveOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;*/

    /*DArray idctNaiveOutput = naiveIDCT(dctNaiveOutput);
    /*std::cout << "Naive IDCT output:\n";
    for (double val : idctNaiveOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;*/

    // Perform fast DCT and IDCT
    /* DArray dctFastOutput = dctFast(input);
    /*std::cout << "Fast DCT output:\n";
    for (double val : dctFastOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;*/

    /*DArray idctFastOutput = idctFast(dctFastOutput);
    /*std::cout << "Fast IDCT output:\n";
    for (double val : idctFastOutput) {
        std::cout << val << " ";
    }
    std::cout << std::endl;*/

    // Convert vectors to CArray for comparison
    /*CArray inputCArray = vectorToCArray(input);
    CArray idctNaiveCArray = vectorToCArray(idctNaiveOutput);
    CArray idctFastCArray = vectorToCArray(idctFastOutput);

    // Compare results with original input
    bool are_equal_naive = are_arrays_equal(inputCArray, idctNaiveCArray);
    bool are_equal_fast = are_arrays_equal(inputCArray, idctFastCArray);

    std::cout << "Naive IDCT result matches original input: " << (are_equal_naive ? "Yes" : "No") << std::endl;
    std::cout << "Fast IDCT result matches original input: " << (are_equal_fast ? "Yes" : "No") << std::endl;

    return 0;

}*/
