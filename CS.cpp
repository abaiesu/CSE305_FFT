#include "CS.h"
#include "dct.h"



// ---------------------- Helpers for the OMP algo --------------------

MatrixXd TwoDDArrayToMatrixXd(const TwoDDArray& twoDArray) {
    int rows = twoDArray.size();
    int cols = twoDArray[0].size();

    MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = twoDArray[i][j]; 
        }
    }

    return matrix;
}

VectorXd DArrayToVectorXd(const DArray& array) {
    int size = array.size();

    VectorXd vector(size);

    for (int i = 0; i < size; ++i) {
        vector(i) = array[i]; 
    }

    return vector;
}

DArray VectorXdToDArray(const VectorXd& vector) {
    int size = vector.size();

    DArray array(size);

    for (int i = 0; i < size; ++i) {
        array[i] = vector(i); // Construct complex numbers from real part of Eigen vector
    }

    return array;
}

TwoDDArray MatrixXdToTwoDDArray(const MatrixXd& matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();

    TwoDDArray twoDArray(rows, DArray(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            twoDArray[i][j] = matrix(i, j); // Construct complex numbers from real part of Eigen matrix
        }
    }

    return twoDArray;
}


// ---------------------------- OMP algo ------------------------------

void get_correlation(const MatrixXd& D, const VectorXd& residual, VectorXd& correlations, int start, int end) {
    for (int i = start; i < end; ++i) {
        double corr = D.col(i).dot(residual);
        //lock_guard<mutex> lock(mtx);
        correlations(i) = corr;
    }
}

void serial_omp (const TwoDDArray& Phi_, const DArray& y_, DArray& x_hat_, int K) {
    

    MatrixXd Phi = TwoDDArrayToMatrixXd(Phi_);
    VectorXd y = DArrayToVectorXd(y_);
    VectorXd x_hat = DArrayToVectorXd(x_hat_);

    int n = Phi.cols();
    VectorXd residual = y;
    IArray indices;
    MatrixXd Phi_j;
    VectorXd coef;

    for (int k = 0; k < K; ++k) {
        
        // Step 1: Find the atom most correlated with the residual
        VectorXd correlations = VectorXd::Zero(n);

        for (int i = 0; i < n; ++i) {
            double corr = Phi.col(i).dot(residual);
            correlations(i) = corr;
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

    }

    // Fill the sparse vector x with the coefficients
    x_hat = VectorXd::Zero(n);
    for (size_t i = 0; i < indices.size(); ++i) {
        x_hat(indices[i]) = coef(i);
    }

    x_hat_ = VectorXdToDArray(x_hat);

}

void parallel_omp (const TwoDDArray& Phi_, const DArray& y_, DArray& x_hat_, int K, int num_threads) {
    

    MatrixXd Phi = TwoDDArrayToMatrixXd(Phi_);
    VectorXd y = DArrayToVectorXd(y_);
    VectorXd x_hat = DArrayToVectorXd(x_hat_);


    int n = Phi.cols();
    VectorXd residual = y;
    IArray indices;
    MatrixXd Phi_j;
    VectorXd coef;

    for (int k = 0; k < K; ++k) {
        
        // Step 1: Find the atom most correlated with the residual
        VectorXd correlations = VectorXd::Zero(n);

        std::vector<std::thread> threads(num_threads);
        int chunk_size = n / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start = t * (n / num_threads);
            int end = (t == num_threads - 1) ? n : (t + 1) * chunk_size;
            threads[t] = std::thread(get_correlation, ref(Phi), ref(residual), ref(correlations), start, end);
        }

        for (auto& thread : threads) {
            thread.join();
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

    }

    // Fill the sparse vector x with the coefficients
    x_hat = VectorXd::Zero(n);
    for (size_t i = 0; i < indices.size(); ++i) {
        x_hat(indices[i]) = coef(i);
    }

    x_hat_ = VectorXdToDArray(x_hat);

}


// ----------------------------- Undersampling ------------------------------


TwoDDArray generate_normal_matrix(int M, int N) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    TwoDDArray matrix(M, DArray(N));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = distribution(gen);
        }
    }
    return matrix;
}

DArray serial_matrix_mul(TwoDDArray& matrix, DArray& vector) {
    
    int rows = matrix.size(); // Number of rows in matrix
    int cols = matrix[0].size(); // Number of columns in matrix

    DArray result(rows, 0); // Resultant vector

    // Perform matrix-vector multiplication
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}

void worker_matrix_mul(const TwoDDArray& A, const DArray& x, DArray& y, int start, int end) {
    
    int cols = A[0].size();

    for (int i = start; i < end; ++i) {
        for (int j = 0; j < cols; ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
}

DArray parallel_matrix_mul(const TwoDDArray& A, const DArray& x, int num_threads){
    int M = A.size();
    DArray y(M, 0.0);
    std::vector<std::thread> threads(num_threads);
    int chunk_size = std::ceil(static_cast<double>(M) / num_threads);

    for (int i = 0, start = 0; i < num_threads; ++i, start += chunk_size) {
        int end = std::min(start + chunk_size, M);
        threads[i] = std::thread(worker_matrix_mul, std::ref(A), std::ref(x), std::ref(y), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return y;
}

