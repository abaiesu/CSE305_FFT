#include "CS.h"
#include "dct.h"
#include "helpers.h"


int main(){

    int num_threads = std::thread::hardware_concurrency();

    printf("Number of threads : %d\n", num_threads);

    std::string filename;

    int N = pow(2, 11);
    int m = N/10;
    printf("Signal length N = %d, measurment size m = %d\n", N, m);

    DArray x = gen_wave(N);
    filename = "x.txt";
    save2txt(x, filename);

    DArray x_dct = dctFast(x);

    TwoDDArray Phi = generate_normal_matrix(m, N);
    DArray y = parallel_matrix_mul(Phi, x_dct, num_threads);
    
    int K = int(N*0.1);

    DArray y_hat(N);
    parallel_omp(Phi, y, y_hat, K, num_threads);

    DArray x_hat = idctFast(y_hat);

    filename = "x_hat.txt";
    save2txt(x_hat, filename);

    return 0;

}