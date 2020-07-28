#include <iostream>
#include <omp.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <unsupported/Eigen/SparseExtra> // For reading MatrixMarket files
#include <unsupported/Eigen/IterativeSolvers>

#include <amgcl/adapter/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/solver/idrs.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/backend/eigen.hpp>

#include "cxxopts.hpp"

int main(int argc, char *argv[])
{

    cxxopts::Options options("SparseSolverBench", "Benchmarking various sparse linear solvers");

    options.add_options()("t,threads", "How many threads to use", cxxopts::value<int>()->default_value("1"))("s,tolerance", "Tolerance to which solvers should converge", cxxopts::value<double>()->default_value("1e-12"))("A,SparseMatrix", "MM Format File for Sparse Matrix", cxxopts::value<std::string>())("b,Rightside", "MM Format File for Vector", cxxopts::value<std::string>())("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    int threads = result["threads"].as<int>();
    double tolerance = result["tolerance"].as<double>();
    std::string A_filename = result["SparseMatrix"].as<std::string>();
    std::string b_filename = result["Rightside"].as<std::string>();

    std::cout << "Running profiler with " << threads << " threads\n"
              << "Required tolerance " << tolerance << "\nA:" << A_filename << "\nb" << b_filename << std::endl;

    omp_set_num_threads(threads);

    amgcl::profiler<> prof;

    // Read sparse matrix from MatrixMarket format.
    // In general this should come pre-assembled.
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;

    prof.tic("read");
    Eigen::loadMarket(A, A_filename);

    Eigen::VectorXd f;
    Eigen::loadMarketVector(f, b_filename);
    prof.toc("read");
    std::vector<double> f2 = std::vector<double>(f.data(), f.data() + f.size());

    // Zero initial approximation:
    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.rows());
    Eigen::VectorXd x3 = x;

    std::vector<double> x2 = std::vector<double>(x.data(), x.data() + x.size());
    std::vector<double> x4 = x2;
    std::vector<double> x5 = x2;
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

    prof.tic("setup_amgcl");
    Solver solve(A_amgcl, prm);
    prof.toc("setup_amgcl");
    std::cout << solve << std::endl;

    // Solve the system for the given RHS:
    int iters;
    double error;
    prof.tic("solve_amgcl");
    std::tie(iters, error) = solve(A_amgcl, f2, x2);
    prof.toc("solve_amgcl");

    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::bicgstabl<amgcl::backend::builtin<double>>>
        Solver_bicgstabl;

    Solver_bicgstabl::params prm_bicgstabl;
    prm_bicgstabl.solver.tol = tolerance;

    prof.tic("setup_amgcl_bicgstabl");
    Solver_bicgstabl solve_bicgstabl(A_amgcl, prm_bicgstabl);
    prof.toc("setup_amgcl_bicgstabl");
    std::cout << solve_bicgstabl << std::endl;

    // Solve the system for the given RHS:
    int iters_bicgstabl;
    double error_bicgstabl;
    prof.tic("solve_amgcl_bicgstabl");
    std::tie(iters_bicgstabl, error_bicgstabl) = solve_bicgstabl(A_amgcl, f2, x4);
    prof.toc("solve_amgcl_bicgstabl");


    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::idrs<amgcl::backend::builtin<double>>>
        Solver_idrs;

    Solver_idrs::params prm_idrs;
    prm_idrs.solver.tol = tolerance;

    prof.tic("setup_amgcl_idrs");
    Solver_idrs solve_idrs(A_amgcl, prm_idrs);
    prof.toc("setup_amgcl_idrs");
    std::cout << solve_idrs << std::endl;

    // Solve the system for the given RHS:
    int iters_idrs;
    double error_idrs;
    prof.tic("solve_amgcl_idrs");
    std::tie(iters_idrs, error_idrs) = solve_idrs(A_amgcl, f2, x5);
    prof.toc("solve_amgcl_idrs");




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
    Solver2 solve2(A, prm2);
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

    prof.tic("setup_eigen_BiCGSTABL");
    Eigen::BiCGSTABL<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_eigen3;
    solver_eigen3.setTolerance(tolerance);
    solver_eigen3.compute(A);
    prof.toc("setup_eigen_BiCGSTABL");
    prof.tic("solve_eigen_BiCGSTABL");
    Eigen::VectorXd result_eigen3 = solver_eigen3.solveWithGuess(f, x3);
    prof.toc("solve_eigen_BiCGSTABL");

    prof.tic("setup_eigen_IDRSTAB");
    Eigen::IDRStab<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_eigen4;
    solver_eigen4.setTolerance(tolerance);
    solver_eigen4.compute(A);
    prof.toc("setup_eigen_IDRSTAB");
    prof.tic("solve_eigen_IDRSTAB");
    Eigen::VectorXd result_eigen4 = solver_eigen4.solveWithGuess(f, x3);
    prof.toc("solve_eigen_IDRSTAB");

    // prof.tic("setup_eigen_LU");
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> solver_eigen2;
    // solver_eigen2.setTolerance(tolerance);
    // solver_eigen2.compute(A);
    //  prof.toc("setup_eigen_LU");
    //  prof.tic("solve_eigen_LU");
    // Eigen::VectorXd result_eigen2 = solver_eigen2.solveWithGuess(f, x3);
    // prof.toc("solve_eigen_LU");
    std::cout<<"Eigen uses "<<Eigen::nbThreads( )<<" threads"<<std::endl;
    std::cout << "amgcl iter:" << iters << " error " << error << std::endl;
    std::cout << "amgcl_bicgstabl iter:" << iters_bicgstabl << " error " << error_bicgstabl << std::endl;
    std::cout << "amgcl_idrs iter:" << iters_idrs << " error " << error_idrs << std::endl;
    std::cout << "amgcl+eigen iter:" << iters2 << " error " << error2 << std::endl;
    std::cout << "eigen iter:" << solver_eigen.iterations() << " error " << solver_eigen.error() << std::endl;
    std::cout << "eigen_bicgstabl iter:" << solver_eigen3.iterations() << " error " << solver_eigen3.error() << std::endl;
    std::cout << "eigen_idrstab iter:" << solver_eigen4.iterations() << " error " << solver_eigen4.error() << std::endl;
    //std::cout << "eigen_LU iter:" << solver_eigen.iterations() << " error " << solver_eigen.error() << std::endl
    std::cout << prof << std::endl;
}