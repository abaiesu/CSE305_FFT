#include "CS.h"
#include "dct.h"
#include "helpers.h"


void cs_demo(){

    int num_threads = 20;
    int M = 4;
    int p = 5;
    ull N = pow(M, p);
    IArray dimensions (p, M);
    int m = N/10;
    printf("Signal length N = %lld, measurment size m = %d\n", N, m);

    std::string filename;

    //------------- Initi -----------
    DArray x = gen_wave(N);
    TwoDDArray Phi = generate_normal_matrix(m, N);
    int K = int(N*0.1);
    DArray y_hat(N);
    filename = "x.txt";
    save2txt(x, filename);


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


    filename = "x_hat.txt";
    save2txt(x_hat, filename);



    /*start = std::chrono::high_resolution_clock::now();

    //-------- Undersampling ---------
    x_dct = serial_dct(x);
    y = serial_matrix_mul(Phi, x_dct);

    //------------- OMP --------------
    serial_omp(Phi, y, y_hat, K);
    x_hat = serial_idct(y_hat);

    end = std::chrono::high_resolution_clock::now();

    elapsed = end - start;

    // Output the time taken
    std::cout << "Serial time: " << elapsed.count() << " sec" << std::endl;*/

}




void dft_demo(){

    int num_threads = 20;
    int M = 4;
    int p = 5;
    ull N = pow(M, p);
    IArray dimensions (p, M);
    
    printf("Signal length N = %lld\n", N);

    std::string filename;

    DArray x = gen_wave(N);
    filename = "x.txt";
    save2txt(x, filename);

    CArray x_c(x.begin(), x.end());
    parallel_dft(x_c, dimensions, num_threads);

    CArray x_sparse = sparsify_data(x_c, N/20); //keep 10% of the coeff

    idft(x_sparse, dimensions, num_threads);

    filename = "x_hat.txt";
    save2txt(x_sparse, filename);

}




int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <0|1>\n", argv[0]);
        fprintf(stderr, "0 : dft_demo\n");
        fprintf(stderr, "1 : cs_demo\n");
        return 1;
    }

    int choice = atoi(argv[1]);
    if (choice == 0) {
        dft_demo();
    } else if (choice == 1) {
        cs_demo();
    } else {
        fprintf(stderr, "Invalid argument. Usage: %s <0|1>\n", argv[0]);
        return 1;
    }

    return 0;
}