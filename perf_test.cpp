#include "dft.h"
#include "CS.h"
#include "helpers.h"
#include "dct.h"

CArray test_1D_fft_diff_num_threads(int M, int p, bool print){

    ull N = pow(M, p); 

    if(print){
        printf("test_num_threads for N = %llu\n", N);
    }

    IArray dimensions (p, M);

    CArray durations(20);

    for (int th = 1; th <= 20; th++){

        int num_threads = 1;

        //give the same data to both algorithms
        CArray data = gen_temp(N);

        // Timing sequential FFT
        auto start = std::chrono::high_resolution_clock::now();
        parallel_dft(data, dimensions, th);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (print){

        }

        durations[th-1] = elapsed.count();

        printf("#threads = %d, duration = %f\n\n", th, elapsed.count());
        
    }

    printf("\n");

    return durations;

}


void test_fft_speed(int M, int p, bool print){

    ull N = pow(M, p); 

    if(print){
        printf("N = %llu\n", N);
    }

    IArray dimensions (p, M);

    CArray x = gen_temp(N);

    int num_threads = 20;

    // Timing sequential FFT
    auto start = std::chrono::high_resolution_clock::now();
    serial_dft(x);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_ser = end - start;

    if (print){
        printf("SERIAL : %f sec\n", elapsed_ser.count());
    }
    
    // Timing sequential FFT
    start = std::chrono::high_resolution_clock::now();
    parallel_dft(x, dimensions, num_threads);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_par = end - start;

    if (print){
        printf("PARALLEL : %f sec\n", elapsed_ser.count());
    }

    printf("Ratio serial/parallel : %f\n", elapsed_ser.count()/elapsed_par.count());

    printf("\n");

       
}


void test_perf_2D_fft(){

    for(int p = 5; p <= 10; p++){

        int N = pow(2, p);
        printf("N = %d\n", N);
        int num_threads = 20;
        TwoDCArray matrix_ser = generate_random_2d_array(N, N, 0, 100);
        TwoDCArray matrix_par = generate_random_2d_array(N, N, 0, 100);


        // Timing sequential FFT
        auto start_sequ = std::chrono::high_resolution_clock::now();
        serial_dft2D(matrix_ser);
        auto end_sequ = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_sequ = end_sequ - start_sequ;

        // Timing parallel FFT
        auto start_par = std::chrono::high_resolution_clock::now();
        parallel_fft2D(matrix_par, num_threads);
        auto end_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_par = end_par - start_par;

        std::cout << "Serial 2D FFT time: " << elapsed_sequ.count() << " seconds" << std::endl;
        std::cout << "Parallel 2D FFT time: " << elapsed_par.count() << " seconds" << std::endl;

        double speedup = elapsed_sequ.count()/elapsed_par.count();
        std::cout << "Speedup: " << speedup << " seconds" << std::endl;

        printf("\n");

    }

}


void test_matrix_mul(int M, int N) {
    
    int num_threads = 20;

    // random Gaussian matrix A
    TwoDDArray A = generate_normal_matrix(M, N);

    // Input vector x
    DArray x = gen_wave(N);


    // Measure time for serial multiplication
    auto start_serial = std::chrono::steady_clock::now();
    DArray serial_result = serial_matrix_vect_mul(A, x);
    auto end_serial = std::chrono::steady_clock::now();
    std::chrono::duration<double> serial_duration = end_serial - start_serial;

    // Measure time for parallel multiplication
    auto start_parallel = std::chrono::steady_clock::now();
    DArray parallel_result = parallel_matrix_mul(A, x, num_threads);
    auto end_parallel = std::chrono::steady_clock::now();
    std::chrono::duration<double> parallel_duration = end_parallel - start_parallel;

    // Output timing results
    std::cout << "Serial multiplication time: " << serial_duration.count() << " seconds" << std::endl;
    std::cout << "Parallel multiplication time: " << parallel_duration.count() << " seconds" << std::endl;

    double speedup = serial_duration.count()/parallel_duration.count();
    std::cout << "Speedup: " << speedup << " seconds" << std::endl;

    if (are_arrays_equal<DArray>(serial_result, parallel_result)) {
        std::cout << "Correct output :)" << std::endl;
    } else {
        std::cout << "Incorrect output :(" << std::endl;
    }

}


void test_omp(){
    
    int num_threads = 20;

    int N = pow(2, 10);
    int M = N/10;
    printf(" N = %d, m = %d\n", N, M);
    int K = int(N*0.05);;
    TwoDDArray A = generate_normal_matrix(M, N);
    DArray y_ser(M);
    DArray y_par(M);
    DArray x = gen_wave(N);
    

    // Time serial_omp
    auto start_serial = std::chrono::steady_clock::now();
    serial_omp(A, y_ser, x, K);
    auto stop_serial = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_serial = stop_serial - start_serial;


    // Time parallel_omp
    auto start_parallel = std::chrono::steady_clock::now();
    parallel_omp(A, y_par, x, K, num_threads);
    auto stop_parallel = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_parallel = stop_parallel - start_parallel;

    // Print the times
    std::cout << "Serial OMP time: " << duration_serial.count() << " seconds\n";
    std::cout << "Parallel OMP time: " << duration_parallel.count() << " seconds\n";

    // Print the ratio
    double ratio = static_cast<double>(duration_serial.count()) / duration_parallel.count();
    std::cout << "Ratio (Serial Time / Parallel Time): " << ratio << endl;

    if (are_arrays_equal<DArray>(y_par, y_ser)) {
        std::cout << "Correct output :)" << std::endl;
    } else {
        std::cout << "Incorrect output :(" << std::endl;
    }

}


void test_dct() {

    int num_threads = 20;

    // Input data
    ull N = pow(2, 15);
    printf("N = %d\n", N);
    int M = 8;
    int p = 5;
    IArray dimensions (p, M);
    DArray input = gen_wave(N);

    // Measure running times
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> out1 = serial_dct(input);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationNaiveIDCT = end - start;
    std::cout << "Serial DCT time: " << durationNaiveIDCT.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    DArray out2 = parallel_dct(input, dimensions, num_threads);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFastDCT = end - start;
    std::cout << "Parallel DCT time: " << durationFastDCT.count() << " seconds\n";

    
    std::cout << "Ratio: " << durationNaiveIDCT.count()/durationFastDCT.count() << " \n";
    
    bool are_equal = are_arrays_equal(out1, out2);

    std::cout << "Correct ? " << (are_equal ? "Yes" : "No") << std::endl;

}



void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <test_number> [args...]\n";
    std::cout << "0: Test 1D FFT num threads\n";
    std::cout << "1: Test FFT speed\n";
    std::cout << "2: Test performance 2D FFT\n";
    std::cout << "3: Test matrix multiplication\n";
    std::cout << "4: Test OMP\n";
    std::cout << "5: Test DCT\n";
}



int main(int argc, char* argv[]) {

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int test_number = std::atoi(argv[1]);

    switch (test_number) {
        case 0:
            if (argc != 5) {
                std::cerr << "Usage: " << argv[0] << " 0 <M> <p> <print>\n";
                return 1;
            }
            {
                int M = std::atoi(argv[2]);
                int p = std::atoi(argv[3]);
                bool print = std::atoi(argv[4]);
                test_1D_fft_diff_num_threads(M, p, print);
            }
            break;

        case 1:
            if (argc != 5) {
                std::cerr << "Usage: " << argv[0] << " 1 <M> <p> <print>\n";
                return 1;
            }
            {
                int M = std::atoi(argv[2]);
                int p = std::atoi(argv[3]);
                bool print = std::atoi(argv[4]);
                test_fft_speed(M, p, print);
            }
            break;

        case 2:
            if (argc != 2) {
                std::cerr << "Usage: " << argv[0] << " 2\n";
                return 1;
            }
            test_perf_2D_fft();
            break;

        case 3:
            if (argc != 4) {
                std::cerr << "Usage: " << argv[0] << " 3 <M> <N>\n";
                return 1;
            }
            {
                int M = std::atoi(argv[2]);
                int N = std::atoi(argv[3]);
                test_matrix_mul(M, N);
            }
            break;

        case 4:
            if (argc != 2) {
                std::cerr << "Usage: " << argv[0] << " 4\n";
                return 1;
            }
            test_omp();
            break;

        case 5:
            if (argc != 2) {
                std::cerr << "Usage: " << argv[0] << " 5\n";
                return 1;
            }
            test_dct();
            break;

        default:
            std::cerr << "Invalid test number\n";
            print_usage(argv[0]);
            return 1;
    }

    return 0;
}