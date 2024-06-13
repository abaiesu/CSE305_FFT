#include "dft.h"
#include "CS.h"

CArray test_1D_fft_num_threads(int M, int p, bool print){

    ull N = pow(M, p); 

    if(print){
        printf("test_num_threads for N = %llu\n", N);
    }

    IArray dimensions (p, M);

    CArray durations(20);

    for (int th = 1; th <= 20; th++){

        int num_threads = 1;

        //give the same data to both algorithms
        CArray data = gen_data(N);

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


void test_matrix_mul() {
    
    int M = 300; 
    int N = 1000; 
    int num_threads = 20;

    // random Gaussian matrix A
    TwoDCArray A = generate_normal_matrix(M, N);

    // Input vector x
    CArray x = gen_data(N);


    // Measure time for serial multiplication
    auto start_serial = std::chrono::steady_clock::now();
    CArray serial_result = serial_matrix_mul(A, x);
    auto end_serial = std::chrono::steady_clock::now();
    std::chrono::duration<double> serial_duration = end_serial - start_serial;

    // Measure time for parallel multiplication
    auto start_parallel = std::chrono::steady_clock::now();
    CArray parallel_result = parallel_matrix_mul(A, x, num_threads);
    auto end_parallel = std::chrono::steady_clock::now();
    std::chrono::duration<double> parallel_duration = end_parallel - start_parallel;

    // Output timing results
    std::cout << "Serial multiplication time: " << serial_duration.count() << " seconds" << std::endl;
    std::cout << "Parallel multiplication time: " << parallel_duration.count() << " seconds" << std::endl;

    double speedup = serial_duration.count()/parallel_duration.count();
    std::cout << "Speedup: " << speedup << " seconds" << std::endl;

    if (are_arrays_equal(serial_result, parallel_result)) {
        std::cout << "Correct output :)" << std::endl;
    } else {
        std::cout << "Incorrect output :(" << std::endl;
    }

}



void test_omp(){
    
    int num_threads = 20;

    int N = 2000;
    int M = 600;
    int K = 100;
    TwoDCArray A = generate_normal_matrix(M, N);
    CArray y_ser(M);
    CArray y_par(M);
    CArray x = gen_data(N);       
    x = sparsify_data(x, K);
    

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
    cout << "Serial OMP time: " << duration_serial.count() << " seconds\n";
    cout << "Parallel OMP time: " << duration_parallel.count() << " seconds\n";

    // Print the ratio
    double ratio = static_cast<double>(duration_serial.count()) / duration_parallel.count();
    cout << "Ratio (Serial Time / Parallel Time): " << ratio << endl;

    if (are_arrays_equal(y_par, y_ser)) {
        std::cout << "Correct output :)" << std::endl;
    } else {
        std::cout << "Incorrect output :(" << std::endl;
    }

}



//int main() {
    
    //test_perf_2D();

    //test_omp();

    
    /*int M, p;
    CArray durations;
    std::string filename;

    M = 2;
    p = 10;
    durations = test_1D_num_threads(M, p, true);
    filename = "dur_M_" + std::to_string(M) + "_p_" + std::to_string(p);
    save2txt(durations, filename);

    int N = pow(M, p);
    CArray sequ = gen_data(N);

    // Timing sequential FFT
    auto start_sequ = std::chrono::high_resolution_clock::now();
    serial_fft(sequ);
    auto end_sequ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sequ = end_sequ - start_sequ;

    std::cout << "Sequential FFT time: " << elapsed_sequ.count() << " seconds" << std::endl;

    /*M = 4;
    p = 5;
    durations = test_num_threads(M, p, true);
    filename = "dur_M_" + std::to_string(M) + "_p_" + std::to_string(p);

    save2txt(durations, filename);*/
    
    /*int M = 4;
    int p = 7;
    ull N = pow(M, p); 

    printf("N = %llu\n", N);

    IArray dimensions (p, M);

    int num_threads = 20;

    //give the same data to both algorithms
    CArray sequ = gen_data(N);
    CArray par = sequ;


    // Timing sequential FFT
    auto start_sequ = std::chrono::high_resolution_clock::now();
    //sequ_fft(sequ);
    parallel_fft(sequ, dimensions, 1);
    auto end_sequ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sequ = end_sequ - start_sequ;

    // Timing parallel FFT
    auto start_par = std::chrono::high_resolution_clock::now();
    parallel_fft(par, dimensions, num_threads);
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_par = end_par - start_par;

    std::cout << "Sequential FFT time: " << elapsed_sequ.count() << " seconds" << std::endl;
    std::cout << "Parallel FFT time: " << elapsed_par.count() << " seconds" << std::endl;

    double speedup = elapsed_sequ.count()/elapsed_par.count();
    std::cout << "Speedup: " << speedup << " seconds" << std::endl;


    //CArray ifft_data = ifft(fft2);  

    if (areArraysEqual(sequ, par)) {
        std::cout << "Correct output :)" << std::endl;
    } else {
        std::cout << "Incorrect output :(" << std::endl;
    }

    //save2txt(fft_data, "fft_data.txt");*/

    

    //return 0;
//}