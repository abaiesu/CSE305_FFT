#include "fft.h"
#include <iostream>
#include <vector>
#include <thread>
#include <Eigen/Dense> 
#include <random>

using namespace std;
using namespace Eigen;


typedef vector<CArray> TwoDCArray;


void fft2D(TwoDCArray & data){

    int rows = data.size();
    int cols = data[0].size();
    
    // fft per row
    for (int i = 0; i < rows; ++i) {
        sequ_fft(data[i]);
    }

    // Transpose the matrix to work on columns using the same 1D FFT
    for (int j = 0; j < cols; ++j) {
        CArray column(rows);
        for (int i = 0; i < rows; ++i) {
            column[i] = data[i][j];
        }

        sequ_fft(column);

        for (int i = 0; i < rows; ++i) {
            data[i][j] = column[i];
        }
    }
}


void correlations(const MatrixXd& D, const VectorXd& residual, VectorXd& correlations, int start, int end) {
    for (int i = start; i < end; ++i) {
        double corr = D.col(i).dot(residual);
        //lock_guard<mutex> lock(mtx);
        correlations(i) = corr;
    }
}


void omp_parallel (const MatrixXd& Phi, const VectorXd& y, VectorXd& x_hat, int K, int num_threads) {
    
    int n = Phi.cols();
    VectorXd residual = y;
    IArray indices;
    MatrixXd Phi_j;
    VectorXd coef;

    for (int k = 0; k < K; ++k) {
        
        // Step 1: Find the atom most correlated with the residual
        VectorXd correlations = VectorXd::Zero(n);
        vector<thread> threads (num_threads);
        //mutex mtx;
        int chunk_size = (n + num_threads - 1) / num_threads;

        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = min(start + chunk_size, n);
            threads[i] = std::thread(correlations, cref(Phi), cref(residual), ref(correlations), start, end);
        }

        for (int i = 0; i < num_threads; ++i) {
            threads[i].join();
        }

        int lambda;
        double max_value = correlations.cwiseAbs().maxCoeff(&lambda);

        // Step 2: Update selected atoms and solve least squares problem
        indices.push_back(lambda);
        Phi_j.resize(y.size(), indices.size());

        for (int i = 0; i < indices.size(); ++i) {
            Phi_j.col(i) = Phi.col(indices[i]);
        }

        coef = Phi_j.jacobiSvd(ComputeThinU | ComputeThinV).solve(y);

        // Step 3: Update the residual
        residual = y - Phi_j * coef;

        // Optional: Break if the residual is small enough
        /*if (residual.norm() < 1e-6) {
            break;
        }*/
    }

    // Fill the sparse vector x with the coefficients
    x_hat = VectorXd::Zero(n);
    for (size_t i = 0; i < indices.size(); ++i) {
        x_hat(indices[i]) = coef(i);
    }

}




/*int main() {


    // Example usage
    MatrixXd D(4, 5);  // Dictionary matrix (4x5)
    D << 1, 0, 0, 1, 0,
         0, 1, 0, 1, 0,
         0, 0, 1, 1, 0,
         1, 1, 1, 1, 1;
    VectorXd y(4);    // Signal vector
    y << 1, 2, 3, 4;
    VectorXd x;       // Sparse representation
    int K = 2; // Desired sparsity level
    int num_threads = 4;

    omp_parallel(D, y, x, K, num_threads);

    cout << "Sparse representation:\n" << x << endl;

    return 0;
}*/

// Function to generate random Gaussian matrix
TwoDCArray generateRandomMatrix(int M, int N) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    TwoDCArray matrix(M, CArray(N));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = distribution(gen);
        }
    }
    return matrix;

}

// Function to perform parallel linear projection
void parallelProjection(TwoDCArray& A, CArray& x, CArray& y, int start, int end) {
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < A.size(); ++j) {
            y[i] += A[j][i] * x[j];
        }
    }
}


int main() {
    
    int M = 20; // Number of rows in matrix A
    int N = 60; // Number of columns in matrix A and size of input vector x

    // Generate random Gaussian matrix A
    TwoDCArray A = generateRandomMatrix(M, N);

    // Input vector x
    CArray x(N, 1.0);

    // Result vector y
    CArray y(M, 0.0);

    // Number of threads to use
    int num_threads = 5;
    std::vector<std::thread> threads(num_threads);

    // Divide the work among threads
    int chunk_size = M / num_threads;
    int start = 0;
    int end = chunk_size;

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        if (i == num_threads - 1) {
            end = M; // Last thread gets remaining work
        }
        threads[i] = std::thread(parallelProjection, std::ref(A), std::ref(x), std::ref(y), start, end);
        start = end;
        end += chunk_size;
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}