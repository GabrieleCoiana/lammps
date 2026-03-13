#include <iostream>
#include <vector>
#include <cmath>

using vector = std::vector<double>;
using matrix = std::vector<std::vector<double>>;

double dot(const vector& a, const vector& b) {
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

vector Ax(const matrix& A, const vector& x) {
    int n = A.size();
    vector result(n, 0.0); // default initialize to 0.0
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    
    return result;
}


int main() {
    int N = 1000000; // size of vectors
    vector a(N, 1.0); // initialize vector a with 1.0
    vector b(N, 2.0); // initialize vector b with 2.0

    double result = dot(a, b);
    std::cout << "Dot product: " << result << std::endl;

    return 0;
}