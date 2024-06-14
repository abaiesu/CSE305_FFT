#include "CS.h"
#include "dct.h"
#include "helpers.h"


int main(){

    int num_threads = std::thread::hardware_concurrency();
    printf("Number of threads : %d\n", num_threads);

    
    ull N = pow(2, 12);
    int M = 8;
    int p = 4;
    IArray dimensions (p, M);
    int m = N/20;
    printf("Signal length N = %lld, measurment size m = %d\n", N, m);

    std::string filename;

    //------------- Initi -----------
    DArray x = gen_wave(N);
    TwoDDArray Phi = generate_normal_matrix(m, N);
    int K = int(N*0.1);
    DArray y_hat(N);
    //filename = "x.txt";
    //save2txt(x, filename);


    auto start = std::chrono::high_resolution_clock::now();
    //-------- Undersampling ---------
    DArray x_dct = parallel_dct(x, dimensions, num_threads);
    DArray y = parallel_matrix_mul(Phi, x_dct, num_threads);

    //------------- OMP --------------
    parallel_omp(Phi, y, y_hat, K, num_threads);
    DArray x_hat = parallel_idct(y_hat, dimensions, num_threads);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    // Output the time taken
    std::cout << "Parallel time: " << elapsed.count() << " sec" << std::endl;


    //filename = "x_hat.txt";
    //save2txt(x_hat, filename);



    start = std::chrono::high_resolution_clock::now();

    //-------- Undersampling ---------
    x_dct = serial_dct(x);
    y = serial_matrix_mul(Phi, x_dct);

    //------------- OMP --------------
    serial_omp(Phi, y, y_hat, K);
    x_hat = serial_idct(y_hat);

    end = std::chrono::high_resolution_clock::now();

    elapsed = end - start;

    // Output the time taken
    std::cout << "Serial time: " << elapsed.count() << " sec" << std::endl;

    return 0;

}