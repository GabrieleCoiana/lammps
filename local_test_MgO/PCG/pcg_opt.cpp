#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>


using vector = std::vector<double>;
using matrix = std::vector<double>; // n*n matrix stored in row-major order

void compare_solutions(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) {
    bool are_equal = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        if (std::abs(x1(i) - x2(i)) > 1e-4) {
            are_equal = false;
            // std::cout << "Difference at index " << i << ": " << std::abs(x1(i) - x2(i)) << "\n";
        }
        max_diff = std::max(max_diff, std::abs(x1(i) - x2(i)));
    }
    if (are_equal) {
        std::cout << "The solutions are approximately equal.\n";
    } else {
        std::cout << "The solutions differ significantly.\n";
    }
    std::cout << "Maximum absolute difference: " << max_diff << "\n";

    double MAE = (x1 - x2).cwiseAbs().mean();
    std::cout << "Mean Absolute Error: " << MAE << "\n";
}

inline double Aij(const matrix& A, int n, int i, int j) {
    return A[(size_t)i * n + j];
}


void Ax_into(const matrix& A, int n, const vector& x, vector& y) {
    // assumes y.size() == n
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const size_t row = (size_t)i * n;
        for (int j = 0; j < n; ++j) {
            s += A[row + j] * x[j];
        }
        y[i] = s;
    }
}


double dot(const vector& a, const vector& b) {
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}



vector pcg(const matrix& A, const vector& b, const vector& M, int max_iter = 1000, double tol = 1e-06) {
    std::cout << "\nStarting custom Preconditioned Conjugate Gradient solver...\n";
    int n = b.size();
    vector x(n, 0.0); // initial guess

    vector r = b; // b - Ax(A, x), but since x is zero, r = b
    
    vector z(n); // preconditioned residual
    vector p(n); // search direction
    vector Ap(n); // A*p
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) z[i] = r[i] / M[i]; // apply preconditioner (Jacobi)   

    p = z; // initial search direction
    
    double rz = dot(r, z); // initial residual magnitude squared
    double r2 = dot(r, r); // for monitoring convergence

    int i = 0;
    while (i < max_iter && r2 > tol * tol)
    {
        Ax_into(A, n, p, Ap); // A*p, O(n^2), better to do this once per iteration, parallel inside
        double pAp = dot(p, Ap);  // p^T A p, parallel inside

        assert (pAp > 0.0);

        double alpha = rz / pAp; // step size that minimizes the error along p


        #pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            x[j] += alpha * p[j]; // update solution
            r[j] -= alpha * Ap[j]; // update residual
            }
        
        // if (i % 100 == 0) {
        //     #pragma omp parallel for
        //     for (int ii = 0; ii < n; ++ii) {
        //         r[ii] = b[ii] - dot(A[ii], x); // recompute residual every 100 iterations to prevent error accumulation
        //     }
        // }

        // update z = M^-1 * r with new residual
        #pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            z[j] = r[j] / M[j]; // apply preconditioner (Jacobi)
        }
      
        double rznew = dot(r, z); // update residual magnitude squared, parallel inside

        double beta = rznew / rz; // coefficient for new search direction

        #pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            p[j] = z[j] + beta * p[j]; // update search direction
        }

    
        rz = rznew;
        r2 = dot(r, r); // for monitoring convergence, parallel inside
        ++i;
    
        

        // // Print progress every 100 iterations
        // if (i % 100 == 0) {
        //     std::cout << "Iteration " << i << ", residual: " << std::sqrt(r2) << "\n";
        // }
    
    }


    if (i == max_iter) {
        std::cout << "CG did not converge within the maximum number of iterations. Final residual: " << std::sqrt(r2) << "\n";
    } else {
        std::cout << "CG converged in " << i << " iterations with residual " << std::sqrt(r2) << "\n";
    }
    return x;
}

int main() {
    // Define matrix dimension
    const int N = 4000;

    // Make a symmetric positive definite matrix A by controlling its eigenvalues
    Eigen::MatrixXd M = Eigen::MatrixXd::Random(N, N);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd Q = qr.householderQ();  

    Eigen::VectorXd eigenvalues(N);
    double lambda_min = 1e-6; // smallest eigenvalue
    double lambda_max = 1.0;  // largest eigenvalue
    double k = lambda_max / lambda_min; // condition number

    for (int i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / (N - 1);
        eigenvalues(i) = lambda_min * std::pow(lambda_max / lambda_min, t);
        // std::cout << "Eigenvalue " << i << ": " << eigenvalues(i) << "\n";
    }
    Eigen::MatrixXd eigen_A = Q * eigenvalues.asDiagonal() * Q.transpose();
    // std::cout << "Condition number of A: " << eigen_A.norm() * eigen_A.inverse().norm() << "\n";

    // Create random b vector
    Eigen::VectorXd eigen_b = Eigen::VectorXd::Random(N); // random b vector

    // // Create NxN matrix
    // matrix A(N, vector(N));
    
    // // Fill with random values
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // auto start_fill = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         A[i][j] = dis(gen);
    //     }
    // }
    // auto end_fill = std::chrono::high_resolution_clock::now();
    // auto duration_fill = std::chrono::duration_cast<std::chrono::milliseconds>(end_fill - start_fill);
    // std::cout << "Time for matrix fill: " << duration_fill.count() << " ms" << std::endl;
    
    // // Symmetrize: A = (A + A^T) / 2
    // for (int i = 0; i < N; ++i) {
    //     for (int j = i; j < N; ++j) {
    //         double avg = (A[i][j] + A[j][i]) / 2.0;
    //         A[i][j] = avg;
    //         A[j][i] = avg;
    //     }
    // }
    
    // // Make positive definite by adding N*I to diagonal
    // auto start_pd = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < N; ++i) {
    //     A[i][i] += 1e-03; // static_cast<double>(N);
    // }
    // auto end_pd = std::chrono::high_resolution_clock::now();
    // auto duration_pd = std::chrono::duration_cast<std::chrono::milliseconds>(end_pd - start_pd);
    // std::cout << "Time for making positive definite: " << duration_pd.count() << " ms" << std::endl;
    
    // // // Print the matrix
    // // std::cout << "Symmetric positive definite matrix:\n";
    // // for (int i = 0; i < N; ++i) {
    // //     for (int j = 0; j < N; ++j) {
    // //         std::cout << A[i][j] << " ";
    // //     }
    // //     std::cout << "\n";
    // // }

    // // Make b vector
    // vector b = vector(N);
    // for (int i = 0; i < N; ++i) {
    //     b[i] = dis(gen);
    // }
    
    
    
    // // Convert your std::vector matrix to Eigen
    // Eigen::MatrixXd eigen_A(N, N);
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         eigen_A(i, j) = A[i][j];
    //     }
    // }
    
    // auto start_det = std::chrono::high_resolution_clock::now();
    // // Compute determinant in one line
    // double det = eigen_A.determinant();
    // std::cout << "Determinant: " << det << "\n";
    // auto end_det = std::chrono::high_resolution_clock::now();
    // auto duration_det = std::chrono::duration_cast<std::chrono::milliseconds>(end_det - start_det);
    // std::cout << "Time for determinant calculation: " << duration_det.count() << " ms" << std::endl;    


    // // creating b vector
    // Eigen::VectorXd eigen_b(N);
    // for (int i = 0; i < N; ++i) {
    //     eigen_b(i) = b[i];
    // }
    

    auto start_solve = std::chrono::high_resolution_clock::now();
    // Solve Ax = b
    // Eigen::VectorXd x = eigen_A.colPivHouseholderQr().solve(eigen_b);
    Eigen::VectorXd x = eigen_A.llt().solve(eigen_b);   // Cholesky (fastest if SPD)
    auto end_solve = std::chrono::high_resolution_clock::now();
    auto duration_solve = std::chrono::duration_cast<std::chrono::milliseconds>(end_solve - start_solve);
    // std::cout << "\nTime for solving Ax = b with Eigen: " << duration_solve.count() << " ms" << std::endl;    

    start_solve = std::chrono::high_resolution_clock::now();
    // Solve with Eigen's Conjugate Gradient
    Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> ecg;
    ecg.setTolerance(1e-6);
    ecg.setMaxIterations(1000);
    ecg.compute(eigen_A);
    Eigen::VectorXd x2 = ecg.solve(eigen_b);

    std::cout << "Eigen CG iters: " << ecg.iterations()
            << " error: " << ecg.error() << "\n";
    end_solve = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::milliseconds>(end_solve - start_solve);
    std::cout << "\nTime for solving Ax = b with Eigen CG: " << duration_solve.count() << " ms" << std::endl;   

    // Compare solutions
    // compare_solutions(x, x2);

    // // Print the solution
    // std::cout << "Solution with Eigen x:\n";
    // for (int i = 0; i < N; ++i) {
    //     std::cout << x(i) << " ";
    // }
    // std::cout << "\n";

    matrix Aflat((size_t)N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Aflat[(size_t)i * N + j] = eigen_A(i, j);

    vector b(N);
    for (int i = 0; i < N; ++i) {
        b[i] = eigen_b(i);
    }

    vector m(N);
    for (int i = 0; i < N; ++i) {        
        m[i] = Aflat[(size_t)i * N + i]; // Jacobi preconditioner
        // m[i] = eigenvalues(i); // ideal preconditioner (eigenvalues of A)
    }




    auto start_cg = std::chrono::high_resolution_clock::now();
    // Solve Ax = b with Conjugate Gradient
    vector x_pcg = pcg(Aflat, b, m);
    // vector x_cg = cg(A, b);
    auto end_cg = std::chrono::high_resolution_clock::now();
    auto duration_cg = std::chrono::duration_cast<std::chrono::milliseconds>(end_cg - start_cg);
    std::cout << "\nTime for solving Ax = b with Conjugate Gradient: " << duration_cg.count() << " ms" << std::endl;
    // // Print the solution
    // std::cout << "Solution with Conjugate Gradient x_cg:\n";
    // for (int i = 0; i < N; ++i) {
    //     std::cout << x_cg[i] << " ";
    // }
    // std::cout << "\n";

    // Compare solutions
    // auto eigen_x_cg = Eigen::Map<Eigen::VectorXd>(x_cg.data(), x_cg.size());
    auto eigen_x_pcg = Eigen::Map<Eigen::VectorXd>(x_pcg.data(), x_pcg.size());

    // std::cout << "Comparing CG solution to Eigen's direct solve:\n";
    // compare_solutions(eigen_x_cg, x);

    // std::cout << "Comparing PCG solution to Eigen's direct solve:\n";
    // compare_solutions(eigen_x_pcg, x);




    return 0;
}