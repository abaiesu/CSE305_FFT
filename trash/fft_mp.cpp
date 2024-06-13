#include "helpers.h"


// ---------------------------------------------------------

// Cooley-Tukey Radix-2 FFT
CArray fft_sequ(CArray& y) {
    
    int n = y.size();
    CArray y_fft(n);
    
    if (n <= 1){
        return y;
    } 

    CArray E(n / 2);
    CArray O(n / 2);
    for (int j = 0; j < n / 2; ++j) {
        E[j] = y[2 * j]; //store the even indices
        O[j] = y[2 * j + 1]; //store the odd indices
    }

    E = fft_sequ(E);
    O = fft_sequ(O);


    Complex W_n = polar(1.0, -2 * PI / n); // W = exp(-2iπ/n)
    Complex W = 1;
    Complex p, q;
    for (int j = 0; j < n / 2; ++j) {
        p = E[j];
        q = W*O[j]; 
        y_fft[j] = p + q;
        y_fft[j + n / 2] = p - q;
        W *= W_n; // update W
    }

    return y_fft;
}


// Inverse FFT using the same FFT function
CArray ifft(CArray& y) {
    
    int n = y.size();
    CArray y_ifft(n);

    // Take the conjugate of the input
    for (int i = 0; i < n; ++i){
        y[i] = std::conj(y[i]);
    }
    
    // Apply forward FFT
    y_ifft = fft_sequ(y);

    // Take the conjugate again
    for (int i = 0; i < n; ++i){
        y_ifft[i] = std::conj(y_ifft[i]) / double(n);
    }

    return y_ifft;
}


//----------------------------------------------------------------


int get_1D_index(IArray &coords, int M, int p) {
    int index = 0;
    for (int i = 0; i < p; ++i) {
        index += pow(M, i - 1) * coords[i];
    }
    return index;
}


IArray get_ND_index(int index, int M, int p ){
     
    IArray indices(p);
    for (int i = 0; i < p; i++){
        indices[i] = index % M;
        index = index / M;
    }
    return indices;
}


void flip(IArray& indices){

    std::reverse(indices.begin(), indices.end());
}


int get_Lj(int j, int M){
    return pow(M, j);    
}

void shiftLeft(std::vector<int>& vec) {
    if (!vec.empty()) {
        std::rotate(vec.begin(), vec.begin() + 1, vec.end());
    }
}


// Generalized FFT (GFFT)
void GFFT(CArray& x, double alpha) {
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
    GFFT(even, alpha);
    GFFT(odd, alpha);

    // Combine
    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = std::polar(1.0, -2 * PI * (k + alpha) / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}


void worker(CArray &input, CArray &res, int start, int stage, int block, int M, int stride, IArray &dimensions) {

    CArray res_temp(M);
    CArray input_temp(M);

    // Load the parts of the input we want to treat
    for (int i = 0; i < M; ++i) {
        input_temp[i] = input[start + i];
    }

    //printf("treating indices from %d to %d\n", start, start + M - 1);

    //#pragma omp parallel for
    
    
    IArray coords = get_ND_index(start, dimensions);

    int g_jm1 = 0;
        
    for (int l = 1; l <= stage - 1; ++l) {
        //printf("l : %d\n", l);  
        int ind = coords.size() - stage + l;
        //printf("ind : %d\n", ind);
        int b = dimensions[coords.size() - stage + l - 1];
        if (l == 1){
            b = 1;
        }
        g_jm1 += coords[coords.size() - stage + l] * b; //get_Lj(l-1, dimensions);       //pow(M, l - 1);
        int a = coords[ind];
        //int b = //get_Lj(l-1, dimensions);

        //printf(" %d (k) x %d (L) + ", a, b);  
        //printf("k_l : %d\n", a);
        //printf("L_l - 1: %d\n\n", b);
    }
    //printf("\n");

    
    double c = get_Lj(stage - 1, dimensions);
    double alpha = (stage == 0) ? 0 : g_jm1 / c;//pow(M, stage - 1);
    //printf("alpha = g_jm1 / L_jm1 =  %d / %.0f = %f\n", g_jm1, c, alpha);

    GFFT(input_temp, alpha);
    res_temp = input_temp;

    for (int k = 0; k < M; ++k) {
        // code inside the loop

        int index = start + k;
        IArray coords = get_ND_index(index, dimensions);

        int k_j = coords.back();

        int g_jm1 = 0;

        for (int l = 1; l <= stage - 1; ++l) {
            //printf("l : %d\n", l);  
            int ind = coords.size() - stage + l;
            //printf("ind : %d\n", ind);*/
            //g_jm1 += coords[coords.size() - stage + l] * pow(M, l - 1);
            int a = coords[ind];
            //printf(" %d ", a);
        }
        //printf("\n");
    }
    

    //#pragma omp parallel for
    //store the result
    for(int i = 0; i < M; i++){
        //printf("inddex %d goes to %d\n", start + i, block + i*stride);
        //printf("old res: %d\n", res[block + i*stride]);
        res[block + i*stride] = res_temp[i];
        //printf("new res: %d\n", res[block + i*stride]);
        //printf("stride : %d, Wrote on index %d\n", stride, block + i*stride);
    }

    //printf("end GFFT\n");
}



CArray parallel_fft(CArray &input, int M, int p) {
    
    int N = input.size();


    if (N != static_cast<int>(pow(M, p))) {
        std::cout << "The size of the input data is not equal to M^p" << std::endl;
        return {};
    }

    CArray res(N);

    // INITIALIZATION
    IArray dimensions(p, M); // {M, M, ..., M} p times
    //IArray dimensions = {2, 4, 2};

    p = dimensions.size();

    //printf("after init\n");

    // 1. Index-reversal permutation
    //#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        IArray indices = get_ND_index(i, dimensions);
        flip(indices);
        int index = get_1D_index(indices, dimensions);
        //printf("new index : %d\n\n", index);  
        res[index] = input[i];
    }

    input = res;

    // 2. For each dimension p
    //int N_i = N / M; // number of blocks
    //int stride = static_cast<int>(pow(M, p - 1)); // stride is always the same
    //printf("stride : %d\n", stride);
    for (int stage = 1; stage <= p; ++stage) {

        //printf("\nSTAGE %d\n", stage);
        
        //#pragma omp parallel for
        /*printf("Before shift\n");
        for (int g = 0; g < dimensions.size(); g++){
            printf(" %d ", dimensions[g]);
        }*/
        if (stage != 1){
            shiftLeft(dimensions);
        }
        /*printf("\nAfter shift\n");
        for (int g = 0; g < dimensions.size(); g++){
            printf(" %d ", dimensions[g]);
        }*/
        
        CArray res(N); //NEW EMPTY RES

        int N_i = N / dimensions[0]; // number of blocks
        std::vector<std::thread> threads (N_i); //number of blocks
        for (int block = 0; block < N_i; ++block) { // for each block
            //printf("\nStage %d BLOCK %d\n", stage, block);
            int start = block * dimensions[0]; //M;

            int stride = N_i; //N/dimensions[stage - 1];

            //printf("GFFT starts\n");
            threads[block] = std::thread(worker, std::ref(input), std::ref(res), start, stage, block, M, stride, std::ref(dimensions));

            //input = res;

            //printf("DONE ONE BLOCK\n");
        }
        for (int i = 0; i < N_i; ++i) {
            threads[i].join();
        }

        input = res;
    }

    return input;
}





bool compareRealPart(const std::complex<double>& a, const std::complex<double>& b) {
    return a.real() > b.real(); // sort in descending order of real part
}




// Keeps only the first num_components components of the data
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
    int p = 10;
    int N = pow(M, p); // Example size of the data
    printf("N : %d\n", N);
    CArray original_data = gen_temp(N);
    CArray original_data_copy = original_data;
    //CArray original_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};//gen_data(N);
    //CArray original_data_copy = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    //CArray original_data = readCSVFile("temperature.csv");
    //original_data.resize(n);
    
    //save2txt(orginal_data, "original_data.txt");

    CArray fft1 = fft_sequ(original_data);
    CArray fft2 = parallel_fft(original_data_copy, M, p);

    printf("good\n");

    //CArray ifft_data = ifft(fft2);  

    if (areArraysEqual(fft1, fft2)) {
        std::cout << "The ifft data is the same as the original data." << std::endl;
    } else {
        std::cout << "The ifft data is different from the original data." << std::endl;
    }

    /*IArray dims = {2, 4, 2};

    for (int i = 0; i < 16; i ++){
        IArray coords = get_ND_index(i, dims);
        printf("index : %d, coords : ", i);
        for (int j = 0; j < coords.size(); j++){
            printf("%d ", coords[j]);
        }
        printf("\n");

    }*/

    //save2txt(fft_data, "fft_data.txt");

    return 0;
}
