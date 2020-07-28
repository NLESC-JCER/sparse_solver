#include <iostream>

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

AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mm>" << std::endl;
        return 1;
    }

    amgcl::profiler<> prof;

    // Read sparse matrix from MatrixMarket format.
    // In general this should come pre-assembled.
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;

    prof.tic("read");
    Eigen::loadMarket(A, argv[1]);
    prof.toc("read");

    // Use vector of ones as RHS for simplicity:
    Eigen::VectorXd f = Eigen::VectorXd::Constant(A.rows(), 1.0);

    // Zero initial approximation:
    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.rows());

    size_t n = A.rows();
const int* ptr = A.outerIndexPtr();
const int* col = A.innerIndexPtr();
const double* val = A.valuePtr();

amgcl::backend::crs<double> A_amgcl(std::make_tuple(n,
		amgcl::make_iterator_range(ptr, ptr + n + 1),
		amgcl::make_iterator_range(col, col + ptr[n]),
		amgcl::make_iterator_range(val, val + ptr[n])
	        ));


    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstab<amgcl::backend::builtin<double> >
        > Solver;

    prof.tic("setup");
    Solver solve(A_amgcl);
    prof.toc("setup");
    std::cout << solve << std::endl;

    // Solve the system for the given RHS:
    int    iters;
    double error;
    prof.tic("solve");
    std::tie(iters, error) = solve(A_amgcl, f, x);
    prof.toc("solve");

    std::cout << iters << " " << error << std::endl
              << prof << std::endl;
}