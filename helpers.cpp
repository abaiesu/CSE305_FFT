#include "helpers.h"

CArray gen_temp(ull n) {
    CArray data(n);
    for (ull i = 0; i < n; ++i) {
        double t = (2 * PI * i) / n;
        data[i] = 20 + 10 * sin(t) + 5 * sin(2 * t) + 2 * ((rand() % 100) / 100.0 - 0.5); 
    }
    return data;
}

DArray gen_wave(ull n) {
    
    std::vector<double> signal(n);
    double pi = 3.14159265358979323846;
    
    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / (n - 1);
        signal[i] = std::cos(2 * 97 * pi * t) + std::cos(2 * 777 * pi * t);
    }
    
    return signal;
}


void save2txt(const CArray& array, const std::string& filename){

    std::ofstream outFile(filename);

    if (outFile.is_open()) {
        // Write the contents of the vector to the file
        for (const auto& complexNum : array) {
            outFile << complexNum.real() << " " << complexNum.imag() << "\n";
        }

        outFile.close();
        std::cout << "Output saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file for writing" << std::endl;
    }

}


void save2txt(const DArray& array, const std::string& filename){

    std::ofstream outFile(filename);

    if (outFile.is_open()) {
        // Write the contents of the vector to the file
        for (const auto& Num : array) {
            outFile << Num << "\n";
        }

        outFile.close();
        std::cout << "Output saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file for writing" << std::endl;
    }

}


CArray readCSVFile(const std::string& filename) {
   
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file." << endl;
        return {};
    }

    // Vector to store complex numbers
    CArray y;

    // Read each line from the file
    string line;
    bool firstLine = true; // Flag to ignore the first line
    while (getline(file, line)) {
        // Ignore the first line
        if (firstLine) {
            firstLine = false;
            continue;
        }
        stringstream ss(line);
        string cell;

        // Vector to store values from a single line
        vector<double> values;

        // Read each value from the line
        while (getline(ss, cell, ',')) {
            // Convert the string value to a double
            double value = stod(cell);
            // Create a complex number with zero imaginary part
            Complex complexNumber(value, 0.0);
            // Add the complex number to the array
            y.push_back(complexNumber);
        }
    }

    // Close the file
    file.close();

    return y;

}

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
        if (std::abs(y_sparse[i]) < std::abs(threash.real())) {
            y_sparse[i] = 0;
        }
    }

    return y_sparse;
}

bool is_power2(int N) {
    return (N > 0) && ((N & (N - 1)) == 0);
}


void read_JPEG(const char* filename, TwoDCArray image, size_t IMAGE_HEIGHT, size_t IMAGE_WIDTH) {
    
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            int pixelValue = fgetc(file);
            image[y][x] = (pixelValue > 127) ? 255 : 0;
        }
    }

    fclose(file);
}



void write_JPEG(const char* filename, const TwoDCArray& image, size_t IMAGE_HEIGHT, size_t IMAGE_WIDTH) {
    
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    // Write JPEG header
    fprintf(file, "P6\n");
    fprintf(file, "%zu %zu\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    fprintf(file, "255\n");

    // Write pixel values
    for (size_t y = 0; y < IMAGE_HEIGHT; y++) {
        for (size_t x = 0; x < IMAGE_WIDTH; x++) {
            // For simplicity, let's just use the real part of the complex number
            int pixelValue = static_cast<int>(image[y][x].real());

            // Ensure the pixel value is within [0, 255]
            pixelValue = max(0, min(255, pixelValue));

            // Write the pixel value as RGB (assuming grayscale)
            fputc(pixelValue, file); // Red
            fputc(pixelValue, file); // Green
            fputc(pixelValue, file); // Blue
        }
    }

    fclose(file);
}




double random_real(double min, double max) {
    return min + static_cast<double>(std::rand()) / RAND_MAX * (max - min);
}


TwoDCArray generate_random_2d_array(int N, int M, double min_value, double max_value) {
    
    TwoDCArray array(N, CArray(M));
    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Fill the array with random real numbers
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            array[i][j] = random_real(min_value, max_value);
        }
    }

    return array;
}


bool are_matrices_equal(TwoDCArray arr1, TwoDCArray arr2){

    // Check if dimensions are the same
    if (arr1.size() != arr2.size() || arr1[0].size() != arr2[0].size()) {
        return false;
    }

    // Check if each corresponding element is equal
    for (size_t i = 0; i < arr1.size(); ++i) {
        for (size_t j = 0; j < arr1[i].size(); ++j) {
            if (std::abs(arr1[i][j] - arr2[i][j]) > 1e-6) {
                return false;
            }
        }
    }

    // If all elements are equal, return true
    return true;
}



