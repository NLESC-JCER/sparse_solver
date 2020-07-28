#include <iostream>
#include <omp.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <unsupported/Eigen/SparseExtra> // For reading MatrixMarket files

#include <amgcl/adapter/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/backend/eigen.hpp>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix.mm>" << std::endl;
        return 1;
    }
    int threads=4;
    double tolerance=1e-12;
    omp_set_num_threads(threads);

    amgcl::profiler<> prof;

    // Read sparse matrix from MatrixMarket format.
    // In general this should come pre-assembled.
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;

    prof.tic("read");
    Eigen::loadMarket(A, argv[1]);
    prof.toc("read");

    // Use vector of ones as RHS for simplicity:
    Eigen::VectorXd f = Eigen::VectorXd::Constant(A.rows(), 1.0);

    std::vector<double> f2 = std::vector<double>(f.data(), f.data() + f.size());

    // Zero initial approximation:
    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.rows());
    Eigen::VectorXd x3 = x;

    std::vector<double> x2 = std::vector<double>(x.data(), x.data() + x.size());

    size_t n = A.rows();
    const int *ptr = A.outerIndexPtr();
    const int *col = A.innerIndexPtr();
    const double *val = A.valuePtr();

    amgcl::backend::crs<double> A_amgcl(std::make_tuple(n,
                                                        amgcl::make_iterator_range(ptr, ptr + n + 1),
                                                        amgcl::make_iterator_range(col, col + ptr[n]),
                                                        amgcl::make_iterator_range(val, val + ptr[n])));

    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::bicgstab<amgcl::backend::builtin<double>>>
        Solver;

Solver::params prm;
prm.solver.tol = tolerance;

    prof.tic("setup");
    Solver solve(A_amgcl,prm);
    prof.toc("setup");
    std::cout << solve << std::endl;

    // Solve the system for the given RHS:
    int iters;
    double error;
    prof.tic("solve");
    std::tie(iters, error) = solve(A_amgcl, f2, x2);
    prof.toc("solve");

    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::eigen<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::bicgstab<amgcl::backend::eigen<double>>>
        Solver2;

Solver2::params prm2;
prm2.solver.tol = tolerance;

    prof.tic("setup_amgcl_eigen");
    Solver2 solve2(A,prm2);
    prof.toc("setup_amgcl_eigen");
    int iters2;
    double error2;
    prof.tic("solve_amgcl_eigen");
    std::tie(iters2, error2) = solve2(A, f, x);
    prof.toc("solve_amgcl_eigen");
    std::cout << solve2 << std::endl;

     prof.tic("setup_eigen");
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_eigen;
    solver_eigen.setTolerance(tolerance); 
    solver_eigen.compute(A);
     prof.toc("setup_eigen");
    prof.tic("solve_eigen");
    Eigen::VectorXd result_eigen = solver_eigen.solveWithGuess(f, x3);
    prof.toc("solve_eigen");

    // prof.tic("setup_eigen_LU");
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> solver_eigen2;
    // solver_eigen2.setTolerance(tolerance); 
    // solver_eigen2.compute(A);
    //  prof.toc("setup_eigen_LU");
    //  prof.tic("solve_eigen_LU");
    // Eigen::VectorXd result_eigen2 = solver_eigen2.solveWithGuess(f, x3);
    // prof.toc("solve_eigen_LU");

    std::cout << "amgcl iter:" << iters << " error " << error << std::endl;
    std::cout << "amgcl+eigen iter:" << iters2 << " error " << error2 << std::endl;
    std::cout << "eigen iter:" << solver_eigen.iterations() << " error " << solver_eigen.error() << std::endl;
    //std::cout << "eigen_LU iter:" << solver_eigen.iterations() << " error " << solver_eigen.error() << std::endl
    std::cout<< prof << std::endl;
}