#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;

typedef complex<double> Complex;
const double PI = acos(-1);

// Generate synthetic periodic weather data (sine wave with noise)
vector<Complex> generateData(int n) {
    vector<Complex> data(n);
    for (int i = 0; i < n; ++i) {
        double t = (2 * PI * i) / n;
        data[i] = 20 + 10 * sin(t) + 5 * sin(2 * t) + 2 * ((rand() % 100) / 100.0 - 0.5); // Simulated temperature data
    }
    return data;
}

// Recursive Cooley-Tukey Radix-2 FFT
void fft(vector<Complex>& P) {
    int n = P.size();
    if (n <= 1) return;

    vector<Complex> U(n / 2);
    vector<Complex> V(n / 2);
    for (int j = 0; j < n / 2; ++j) {
        U[j] = P[2 * j];
        V[j] = P[2 * j + 1];
    }

    fft(U);
    fft(V);

    Complex omega_n = polar(1.0, -2 * PI / n); // ω_n = cos(2π/n) + i*sin(2π/n)
    Complex omega = 1;
    for (int j = 0; j < n / 2; ++j) {
        P[j] = U[j] + omega * V[j];
        P[j + n / 2] = U[j] - omega * V[j];
        omega *= omega_n; // Update ω
    }
}

// Inverse FFT using the same FFT function
void ifft(vector<Complex>& a) {
    for (auto& x : a) x = conj(x);
    fft(a);
    for (auto& x : a) x = conj(x) / double(a.size());
}

// Approximate the signal using the most significant frequencies
vector<Complex> approximateSignal(vector<Complex>& data, int num_components) {
    int n = data.size();
    vector<Complex> freq_data = data;

    // Zero out all but the first num_components/2 and last num_components/2 frequencies
    for (int i = num_components / 2; i < n - num_components / 2; ++i) {
        freq_data[i] = 0;
    }

    ifft(freq_data);

    return freq_data;
}

// Output data to a file
void outputData(const vector<Complex>& original, const vector<Complex>& approximated, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file for writing." << endl;
        return;
    }

    file << "Original,Approximated\n";
    for (size_t i = 0; i < original.size(); ++i) {
        file << original[i].real() << "," << approximated[i].real() << "\n";
    }
    file.close();
}

int main() {
    srand(time(0));

    int n = 1024; // Number of data points
    vector<Complex> original_data = generateData(n);

    // Apply FFT
    vector<Complex> data = original_data;
    fft(data);

    // Approximate the signal using the first few frequency components
    int num_components = 20; // Number of frequency components to keep
    vector<Complex> approximated_data = approximateSignal(data, num_components);

    // Output data to file
    outputData(original_data, approximated_data, "fft1_data.csv");

    return 0;
}
