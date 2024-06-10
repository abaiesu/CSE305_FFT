#include "helpers.h"


// -------------------------------- SEQUENTIAL --------------------------------

void sequ_fft(CArray& y) {
    
    int n = y.size();
    
    if (n <= 1){
        return;
    } 

    CArray E(n / 2);
    CArray O(n / 2);
    for (int j = 0; j < n / 2; ++j) {
        E[j] = y[2 * j]; //store the even indices
        O[j] = y[2 * j + 1]; //store the odd indices
    }

    sequ_fft(E);
    sequ_fft(O);


    Complex W_n = polar(1.0, -2 * PI / n); // W = exp(-2iπ/n)
    Complex W = 1;
    Complex p, q;
    for (int j = 0; j < n / 2; ++j) {
        p = E[j];
        q = W*O[j]; 
        y[j] = p + q;
        y[j + n / 2] = p - q;
        W *= W_n; // update W
    }
}


// -------------------------------- PARALLEL --------------------------------


int get_1D_index(IArray &coords, IArray &dimensions) {
    int index = 0;
    for (int i = 0; i < dimensions.size(); ++i) {
        int partial = 1;
        for(int j = 0; j < i; ++j) {
            partial *= dimensions[j];
        }
        index += partial * coords[i];
    }
    return index;
}


IArray get_ND_index(int index, IArray& dimensions){
    int p = dimensions.size(); 
    IArray indices(p);
    for (int i = 0; i < p; i++){
        indices[i] = index % dimensions[i];
        index = index / dimensions[i];
    }
    return indices;
}


void flip(IArray& indices){
    std::reverse(indices.begin(), indices.end());
}


int get_Lj(int j, IArray &dimensions){
    if(j == 0){
        return 1;
    }
    IArray flipped = dimensions;
    flip(flipped);
    int Lj = 1;
    for (int i = 0; i < j; ++i){
        Lj *= flipped[i];
    }
    return Lj;
}


void shiftLeft(std::vector<int>& vec) {
    if (!vec.empty()) {
        std::rotate(vec.begin(), vec.begin() + 1, vec.end());
    }
}


void gfft(CArray& x, double alpha) {
    const size_t N = x.size();
    if (N <= 1) return;

    // Divide
    CArray even(N / 2);
    CArray odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Conquer
    gfft(even, alpha);
    gfft(odd, alpha);

    // Combine
    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = std::polar(1.0, -2 * PI * (k + alpha) / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}


void block_worker(CArray &input, CArray &res, int start, int stage, int block_index, int block_size, int stride, IArray &dimensions) {

    CArray res_temp(block_size);
    CArray input_temp(block_size);

    // Load the parts of the input we want to treat
    for (int i = 0; i < block_size; ++i) {
        input_temp[i] = input[start + i];
    }
    
    
    IArray coords = get_ND_index(start, dimensions);

    int g_jm1 = 0;
        
    for (int l = 1; l <= stage - 1; ++l){
        int k_index = coords.size() - stage + l;
        IArray dims = dimensions;
        flip(dims);
        int L_term = get_Lj(l-1, dims);
        int K_term = coords[k_index];
        g_jm1 += K_term * L_term; 
    }

    
    double L  = get_Lj(stage - 1, dimensions);
    double alpha = (stage == 0) ? 0 : g_jm1 / L;

    gfft(input_temp, alpha);
    res_temp = input_temp;
    
    //#pragma omp parallel for
    //store the result
    for(int i = 0; i < block_size; i++){
        res[block_index + i*stride] = res_temp[i];
    }
}


void index_reversal_worker(CArray &input, CArray &res, int i, IArray &dimensions) {
    IArray indices = get_ND_index(i, dimensions);
    flip(indices);
    int index = get_1D_index(indices, dimensions);
    res[index] = input[i];
}


void parallel_fft(CArray &input, IArray &dimensions) {
    
    int N = input.size();
    CArray inter_res(N); //store the intermediae result
    int p = dimensions.size();

    // 1. Index-reversal permutation (in parallel)

    std::vector<std::thread> threads (N);
    for (int i = 0; i < N; ++i){
        threads[i] = std::thread(index_reversal_worker, std::ref(input), std::ref(inter_res), i, std::ref(dimensions));
    }

    for (int i = 0; i < N; ++i) {
        threads[i].join();
    }

    input = inter_res;

    // 2. For each dimension p
    for (int stage = 1; stage <= p; ++stage) {

        /* after the first stage, we shift the dimensions to the left
        ex :    input dimension : 2 x 4 x 2
                stage 1 = 2 x 4 x 2
                stage 2 = 4 x 2 x 2
                stage 3 = 2 x 2 x 4
        */
        if (stage != 1){ 
            shiftLeft(dimensions);
        }
    
        CArray inter_res(N); //get a new array to store the intermediate vector

        int num_blocks = N / dimensions[0]; // number of blocks
        int block_size = dimensions[0]; // size of each block
        std::vector<std::thread> threads (num_blocks); //one thread per block
        for (int block_index = 0; block_index < num_blocks; ++block_index) { // for each block (in parallel)
            int start = block_index * dimensions[0]; // starting index of the block
            threads[block_index] = std::thread(block_worker, std::ref(input), std::ref(inter_res), 
                                                        start, stage, block_index, block_size, 
                                                        num_blocks, std::ref(dimensions));
        }
        
        for (int i = 0; i < num_blocks; ++i) {
            threads[i].join();
        }

        input = inter_res;
    }

}

// -------------------------------- Inverse FFT --------------------------------

void ifft(CArray& y, IArray &dimensions) {
    
    int n = y.size();
    CArray y_ifft(n);

    // Take the conjugate of the input
    for (int i = 0; i < n; ++i){
        y[i] = std::conj(y[i]);
    }
    
    if (dimensions.size() != 0){
        parallel_fft(y, dimensions);
    } else {
        sequ_fft(y);
    }

    // Take the conjugate again
    for (int i = 0; i < n; ++i){
        y[i] = std::conj(y[i]) / double(n);
    }
}

// -----------------------------------------------------------------------------


bool compareRealPart(const std::complex<double>& a, const std::complex<double>& b) {
    return a.real() > b.real(); // sort in descending order of real part
}


CArray sparsify_data(CArray& y, int num_components) {
    
    int n = y.size();
    CArray y_sparse = y;
    CArray y_sorted = y;

    std::sort(y_sorted.begin(), y_sorted.end(), compareRealPart);

    Complex threash = y_sorted[num_components - 1];

    for (int i = 0; i < n; ++i) {
        if (y_sparse[i].real() < threash.real()) {
            y_sparse[i] = 0;
        }
    }

    return y_sparse;
}


int main() {
    
    int M = 2;
    int p = 4;
    int N = pow(M, p); // Example size of the data
    printf("N : %d\n", N);
    CArray data1 = gen_data(N);
    CArray data2 = data1;

    CArray original_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};//gen_data(N);
    CArray original_data_copy = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    //CArray original_data = readCSVFile("temperature.csv");
    //original_data.resize(n);
    
    //save2txt(orginal_data, "original_data.txt");

    //sequ_fft(original_data);
    IArray dimensions (M, p);
    parallel_fft(original_data_copy, dimensions);
    ifft(original_data_copy, dimensions);

    //CArray ifft_data = ifft(fft2);  

    if (areArraysEqual(original_data, original_data_copy)) {
        std::cout << "The ifft data is the same as the original data." << std::endl;
    } else {
        std::cout << "The ifft data is different from the original data." << std::endl;
    }

    //save2txt(fft_data, "fft_data.txt");

    return 0;
}
