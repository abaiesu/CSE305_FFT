#include "dct.h"


// -------------------------- Naive DCT-II and inverse -----------------------------

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


DArray naiveIDCT(const DArray & input) {
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


//-------------------------- DCT-II based of DFT SERIAL -----------------------------

DArray serial_dct(const DArray& input) {
    
    int N = input.size();
    DArray v(N, 0.0);

    // Prepare the input for FFT
    for (int i = 0; i <= (N - 1) / 2; ++i) {
        v[i] = input[2 * i];
    }


    for (int i = 0; i < N / 2; ++i) {
        v[N / 2 + i] = input[N - 1 - 2 * i];
    }

    CArray V(v.begin(), v.end());
    serial_dft(V);

    // Compute the DCT-II result
    DArray result(N);
    for (int k = 0; k < N; ++k) {
        std::complex<double> factor = std::exp(std::complex<double>(0, -PI * k / (2.0 * N))) * 2.0;
        if (N % 2 != 0 && k == 0) {
            factor /= std::sqrt(2);
        }
        result[k] = (V[k] * factor).real()*std::sqrt(1.0 / (2.0 * N));
    }

    // Scale the output
    result[0] *= double(1/std::sqrt(2));

    return result;
}


//-------------------------- DCT-II based of DFT PARALLEL -----------------------------

void worker_dct(CArray& V, DArray& result, int start, int end){
    
    int N = result.size();
    for (int k = start; k < end; ++k) {
        Complex factor = std::exp(std::complex<double>(0, -PI * k / (2.0 * N))) * 2.0;
        if (N % 2 != 0 && k == 0) {
            factor /= std::sqrt(2);
        }
        result[k] = (V[k] * factor).real()*std::sqrt(1.0 / (2.0 * N));
    }
}


DArray parallel_dct(const DArray& input, IArray dimensions, int num_threads) {
    
    int N = input.size();
    CArray v(N);

    for (int i = 0; i <= (N - 1) / 2; ++i) {
        v[i] = input[2 * i];
        v[N / 2 + i] = input[N - 1 - 2 * i];
    }

    parallel_dft(v, dimensions, num_threads);

    // Compute the DCT-II result
    DArray result(N);
    std::vector<std::thread> threads (num_threads);
    int num_blocks = N/num_threads;
    int start, end;
    for (int i = 0; i < num_threads; ++i) {
        start = i * num_blocks;
        end = (i == num_threads - 1) ? N : (i + 1) * num_blocks;
        threads[i] = std::thread(worker_dct, std::ref(v), std::ref(result), start, end);
    }

    for (int i = 0; i < num_threads; ++i){
        threads[i].join();
    }

    // Scale the output
    result[0] *= double(1/std::sqrt(2));

    return result;
}



//----------------------- Inverse DCT-II based of DFT SERIAL --------------------------- 

DArray serial_idct(const DArray& input) {
    
    int N = input.size();
    CArray v(N);

    for (int i = 0; i < N; ++i) {
        v[i] = std::sqrt(2.0 * N) * std::exp(std::complex<double>(0, PI * i / (2.0 * N)));
    }
    v[0] /= std::sqrt(2.0);

    CArray temp(N);
    for (int i = 0; i < N; ++i) {
        temp[i] = v[i] * input[i];
    }

    // Perform IFFT
    IArray dimensions;
    int num_threads = 1;
    idft(temp, dimensions, num_threads);

    // Reconstruct the original signal
    DArray x(N);
    for (int i = 0; i < N / 2; ++i) {
        x[2 * i] = temp[i].real();
        x[2 * i + 1] = temp[N - i - 1].real();
    }

    return x;
}


//----------------------- Inverse DCT-II based of DFT PARALLEL ------------------------- 

void worker_idct(DArray& input, CArray& result, int start, int end){
    int N = input.size();
    for (int i = start; i < end; ++i) {
        Complex alpha = std::sqrt(2.0 * N) * std::exp(std::complex<double>(0, PI * i / (2.0 * N)));
        result[i] = alpha * input[i];
    }
}


DArray parallel_idct(DArray& input, IArray dimensions, int num_threads) {
    
    int N = input.size();
    CArray result(N);

    std::vector<std::thread> threads (num_threads);
    int num_blocks = N/num_threads;
    int start, end;
    for (int i = 0; i < num_threads; ++i) {
        start = i * num_blocks;
        end = (i == num_threads - 1) ? N : (i + 1) * num_blocks;
        threads[i] = std::thread(worker_idct, std::ref(input), std::ref(result), start, end);
    }

    for (int i = 0; i < num_threads; ++i){
        threads[i].join();
    }

    result[0] /= std::sqrt(2.0);

    // Perform IFFT
    idft(result, dimensions, num_threads);

    // Reconstruct the original signal
    DArray x(N);
    for (int i = 0; i < N / 2; ++i) {
        x[2 * i] = result[i].real();
        x[2 * i + 1] = result[N - i - 1].real();
    }

    return x;
}




