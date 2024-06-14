#ifndef HELPERS_H
#define HELPERS_H


#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <thread>
#include <mutex>

using namespace std;

typedef complex<double> Complex;
typedef vector<Complex> CArray;
typedef vector<int> IArray;
typedef vector<double> DArray;
typedef unsigned long long int ull;
typedef vector<CArray> TwoDCArray;
typedef vector<DArray> TwoDDArray;
const double PI = acos(-1);


CArray gen_temp(ull n);

std::vector<double> gen_wave(ull n);

void save2txt(const CArray& array, const std::string& filename);

void save2txt(const DArray& array, const std::string& filename);

CArray readCSVFile(const std::string& filename);


template <typename T>
bool are_arrays_equal(const T& a, const T& b) {
    
    double tolerance = 1e-4;
    
    if (a.size() != b.size()) {
        printf("no same sizes\n");
        return false;
    }

    bool flag = true;
    for (size_t i = 0; i < a.size(); ++i) {
        //printf("a[%zu] = (%f, %f), b[%zu] = (%f, %f)\n", i, a[i].real(), a[i].imag(), i, b[i].real(), b[i].imag());
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return flag;
}





bool compareRealPart(const std::complex<double>& a, const std::complex<double>& b);

CArray sparsify_data(CArray& y, int num_components);

void read_JPEG(const char* filename, TwoDCArray image, size_t IMAGE_HEIGHT, size_t IMAGE_WIDTH);

void write_JPEG(const char* filename, const TwoDCArray& image, size_t IMAGE_HEIGHT, size_t IMAGE_WIDTH);

bool is_power2(int N);

TwoDCArray generate_random_2d_array(int N, int M, double min_value, double max_value);

bool are_matrices_equal(TwoDCArray arr1, TwoDCArray arr2);

#endif // HELPERS_H