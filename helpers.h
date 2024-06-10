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
const double PI = acos(-1);

CArray gen_data(int n);
void save2txt(CArray array, std::string filename);
CArray readCSVFile(const std::string& filename) ;
bool areArraysEqual(const CArray& a, const CArray& b);