#include "dft.h"


// -------------------------------- SEQUENTIAL --------------------------------

void serial_dft(CArray& y) {
    
    int n = y.size();

    if (is_power2(n) == false){
        std::cerr << "The size of the input isn't a power of 2" << std::endl;
        return;
    }
    
    if (n <= 1){
        return;
    } 

    CArray E(n / 2);
    CArray O(n / 2);
    for (int j = 0; j < n / 2; ++j) {
        E[j] = y[2 * j]; //store the even indices
        O[j] = y[2 * j + 1]; //store the odd indices
    }

    serial_dft(E);
    serial_dft(O);


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

    if (N <= 1){
        return;
    }

    if (N == 2){ //Radix 2 = Chunks of 2
        Complex even = x[0];
        Complex odd = x[1];
        Complex Twiddle = std::polar(1.0, -2 * PI * alpha / N);
        Complex t = Twiddle * odd;
        x[0] = even + t;
        x[1] = even - t;
        return;
    }

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
    Complex W_N = std::polar(1.0, -2 * PI / N);
    Complex W = 1;
    Complex W_alpha = std::polar(1.0, -2 * PI * alpha / N);
    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = W * W_alpha * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
        W *= W_N;
    }
}



void block_worker(CArray &input, CArray &res, int starting_block_index, int stage, 
                    int num_blocks_to_process, int block_size, int stride, 
                    IArray &dimensions, IArray &L_js) {

    
    
    for(int q = 0; q < num_blocks_to_process; q++){

        int block_index = starting_block_index + q;
        int start = block_index * block_size; 

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
            //IArray dims = dimensions;
            //flip(dims);
            //int L_term = get_Lj(l-1, dims);
            int L_term = L_js[l-1];
            int K_term = coords[k_index];
            g_jm1 += K_term * L_term; 
        }

        
        double L  = L_js[stage - 1];//get_Lj(stage - 1, dimensions);
        double alpha = (stage == 0) ? 0 : g_jm1 / L;

        gfft(input_temp, alpha);
        
        //store the result
        for(int i = 0; i < block_size; i++){
            res[block_index + i*stride] = input_temp[i];
        }
    }

}


void index_reversal_worker(CArray &input, CArray &res, int start, int end, IArray &dimensions) {
    
    for(int i = start; i < end; i++){
        // Get the N-dimensional index (i.e. the coordinates)
        IArray indices = get_ND_index(i, dimensions);
        flip(indices);
        int index = get_1D_index(indices, dimensions);
        res[index] = input[i];
    }
}


void parallel_dft(CArray &input, IArray &dimensions, int num_threads) {
    
    int N = input.size();
    int sum_dims = get_Lj(dimensions.size(), dimensions);

    if (N != sum_dims){
        std::cerr << "The size of the input array is not equal to the product of the dimensions" << std::endl;
        return;
    }


    CArray inter_res(N); //store the intermediae result
    int p = dimensions.size();

    // 1. Index-reversal permutation (in parallel)

    int working_threads = (num_threads > N) ? N : num_threads;
    std::vector<std::thread> threads (working_threads);
    int num_clients = N / working_threads;
    for (int i = 0; i < working_threads; ++i){
        int start = i * num_clients;
        int end = (i == working_threads - 1) ? N : (i + 1) * num_clients;
        threads[i] = std::thread(index_reversal_worker, std::ref(input), std::ref(inter_res), start, end, std::ref(dimensions));
    }

    for (int i = 0; i < working_threads; ++i) {
        threads[i].join();
    }

    input = inter_res;

    // Precompute the L_js
    IArray L_js(p + 1);
    L_js[0] = 1;
    for (int j = 1; j <= p; ++j){
        L_js[j] = pow(dimensions[0], j); 
    }

    // 2. For each dimension p
    int num_blocks = N / dimensions[0]; // number of blocks
    working_threads = (num_threads > num_blocks) ? num_blocks : num_threads;
    int num_blocks_per_thread = num_blocks / working_threads;
    int block_size = dimensions[0]; // size of each block
    for (int stage = 1; stage <= p; ++stage) {

        /* after the first stage, we shift the dimensions to the left
        ex :    input dimension : 2 x 4 x 2
                stage 1 = 2 x 4 x 2
                stage 2 = 4 x 2 x 2
                stage 3 = 2 x 2 x 4
        */
        
        /*if (stage != 1){ 
            shiftLeft(dimensions);
        }
        
        ONLY RELEVANT WHEN THE Ns ARENT ALL THE SAME
        */
    
        CArray inter_res(N); //get a new array to store the intermediate vector
        std::vector<std::thread> threads (working_threads); //one thread per block

        for (int thread_index = 0; thread_index < working_threads; ++thread_index) { // for each block (in parallel)
            
            int num_blocks_to_process = (thread_index == working_threads - 1) ? num_blocks_per_thread + num_blocks % working_threads : num_blocks_per_thread;
            int starting_block_index = thread_index * num_blocks_per_thread;
            threads[thread_index] = std::thread(block_worker, std::ref(input), std::ref(inter_res), 
                                                        starting_block_index, stage,
                                                        num_blocks_to_process, block_size, 
                                                        num_blocks, std::ref(dimensions),
                                                        std::ref(L_js));

        
        }
        
        for (int i = 0; i < working_threads; ++i) {
            threads[i].join();
        }


        input = inter_res;
    }

}

// -------------------------------- Inverse FFT --------------------------------

void idft(CArray& y, IArray &dimensions, const int num_threads = 5) {
    
    int n = y.size();
    CArray y_ifft(n);

    // Take the conjugate of the input
    for (int i = 0; i < n; ++i){
        y[i] = std::conj(y[i]);
    }
    
    if (dimensions.size() != 0){
        parallel_dft(y, dimensions, num_threads);
    } else {
        serial_dft(y);
    }

    // Take the conjugate again
    for (int i = 0; i < n; ++i){
        y[i] = std::conj(y[i]) / double(n);
    }
}

// -------------------------------- Serial 2D FFT ---------------------------------


void serial_dft2D(TwoDCArray & data){

    int nb_rows = data.size();
    int nb_cols = data[0].size();
    
    // fft per row

    for (int i = 0; i < nb_rows; ++i) {
        serial_dft(data[i]);
    }

    // transpose the matrix 
    for (int j = 0; j < nb_cols; ++j) {
        CArray column(nb_rows);
        for (int i = 0; i < nb_rows; ++i) {
            column[i] = data[i][j];
        }

        serial_dft(column);

        for (int i = 0; i < nb_rows; ++i) {
            data[i][j] = column[i];
        }
    }
}


// -------------------------------- Parallel 2D FFT ---------------------------------


void rows_worker_dft2D(TwoDCArray &data, int start_index, int end_index){
    
    for (int i = start_index; i < end_index; ++i) {
        serial_dft(data[i]);
    }

}


void cols_worker_dft2D(TwoDCArray &data, int start_index, int end_index){
    
    int nb_rows = data.size();
    for (int j = start_index; j < end_index; ++j) {
        CArray column(nb_rows);
        for (int i = 0; i < nb_rows; ++i) {
            column[i] = data[i][j];
        }
        serial_dft(column);
        for (int i = 0; i < nb_rows; ++i) {
            data[i][j] = column[i];
        }
    }

}


void parallel_fft2D(TwoDCArray & data, int num_threads){

    int nb_rows = data.size();
    int nb_cols = data[0].size();

    int start_index, end_index, block_size;
    
    // fft per row
    std::vector<std::thread> threads (num_threads);
    block_size = nb_rows / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        start_index = i*block_size,
        end_index = (i == num_threads - 1) ? nb_rows : (i + 1) * block_size;
        threads[i] = std::thread(rows_worker_dft2D, std::ref(data), start_index, end_index);
    }

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    // transpose the matrix 
    block_size = nb_cols / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        start_index = i*block_size,
        end_index = (i == num_threads - 1) ? nb_cols : (i + 1) * block_size;
        threads[i] = std::thread(cols_worker_dft2D, std::ref(data), start_index, end_index);
    }

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

}




