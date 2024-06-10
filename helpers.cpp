#include "helpers.h"

// Generate synthetic periodic weather data (sine wave with noise)
CArray gen_data(int n) {
    CArray data(n);
    for (int i = 0; i < n; ++i) {
        double t = (2 * PI * i) / n;
        data[i] = 20 + 10 * sin(t) + 5 * sin(2 * t) + 2 * ((rand() % 100) / 100.0 - 0.5); // Simulated temperature data
    }
    return data;
}


void save2txt(CArray array, std::string filename) {

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


bool areArraysEqual(const CArray& a, const CArray& b) {
    
    double tolerance = 1e-4;
    
    if (a.size() != b.size()) {
        printf("no same sizes\n");
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        //printf("a[%zu] = (%f, %f), b[%zu] = (%f, %f)\n", i, a[i].real(), a[i].imag(), i, b[i].real(), b[i].imag());
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
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
