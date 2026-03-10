#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

int main() {
    // Define matrix dimension
    const int N = 5;
    
    // Create NxN matrix
    std::vector<std::vector<double>> A(N, std::vector<double>(N));
    
    // Fill with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = dis(gen);
        }
    }
    
    // Symmetrize: A = (A + A^T) / 2
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            double avg = (A[i][j] + A[j][i]) / 2.0;
            A[i][j] = avg;
            A[j][i] = avg;
        }
    }
    
    // Make positive definite by adding N*I to diagonal
    for (int i = 0; i < N; ++i) {
        A[i][i] += static_cast<double>(N);
    }
    
    // Print the matrix
    std::cout << "Symmetric positive definite matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << "\n";
    }
    
    // Convert your std::vector matrix to Eigen
    Eigen::MatrixXd eigen_A(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            eigen_A(i, j) = A[i][j];
        }
    }

    // Compute determinant in one line
    double det = eigen_A.determinant();
    std::cout << "Determinant: " << det << "\n";
    
    // creating b vector
    Eigen::VectorXd b(N);
    for (int i = 0; i < N; ++i) {
        b(i) = dis(gen);
    }

    // Solve Ax = b
    Eigen::VectorXd x = eigen_A.colPivHouseholderQr().solve(b);
    
    // Print the solution
    std::cout << "Solution with Eigen x:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << x(i) << " ";
    }
    std::cout << "\n";

    return 0;
}