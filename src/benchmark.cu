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
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/backend/eigen.hpp>

#include <amgcl/backend/cuda.hpp>
#include <amgcl/relaxation/cusparse_ilu0.hpp>

#include "cxxopts.hpp"

struct LinearSystem
{
    Eigen::VectorXd b;
    Eigen::VectorXd x0;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
};

struct Convoptions
{
    int iterations;
    double tolerance;
};

struct returnvalue
{
    int iterations;
    double error;
    double error_exact;
};

template <class T>
returnvalue RunEigen_Solver(amgcl::profiler<> &prof, Convoptions opt, const LinearSystem &Axb, const std::string &name)
{

    prof.tic("setup_" + name);
    T Solver;
    Solver.setTolerance(opt.tolerance);
    Solver.setMaxIterations(opt.iterations);
    Solver.compute(Axb.A);
    prof.toc("setup_" + name);
    prof.tic("solve_" + name);
    Eigen::VectorXd result_eigen = Solver.solveWithGuess(Axb.b, Axb.x0);
    prof.toc("solve_" + name);
    returnvalue result;
    result.iterations = Solver.iterations();
    result.error = Solver.error();
    result.error_exact=(Axb.A*result_eigen-Axb.b).norm()/Axb.b.norm();
    return result;
}

template <class T>
returnvalue RunAMGCLEigen_backend(amgcl::profiler<> &prof, Convoptions opt, const LinearSystem &Axb, const std::string &name)
{

    Eigen::VectorXd x = Axb.x0;
    typename T::params prm;
    prm.solver.tol = opt.tolerance;
    prm.solver.maxiter = opt.iterations;
    prof.tic("setup_" + name);
    T solve(Axb.A, prm);
    prof.toc("setup_" + name);
    returnvalue result;
    prof.tic("solve_" + name);
    std::tie(result.iterations, result.error) = solve(Axb.A, Axb.b, x);
    prof.toc("solve_" + name);
    std::cout << solve << std::endl;
    result.error_exact=(Axb.A*result_eigen-Axb.b).norm()/Axb.b.norm();
    return result;
}

template <class T>
returnvalue RunAMGCL_backend(amgcl::profiler<> &prof, Convoptions opt, const LinearSystem &Axb, const std::string &name)
{

    std::vector<double> x0 = std::vector<double>(Axb.x0.data(), Axb.x0.data() + Axb.x0.size());
    std::vector<double> b = std::vector<double>(Axb.b.data(), Axb.b.data() + Axb.b.size());
    size_t n = Axb.A.rows();
    const int *ptr = Axb.A.outerIndexPtr();
    const int *col = Axb.A.innerIndexPtr();
    const double *val = Axb.A.valuePtr();
    amgcl::backend::crs<double> A_amgcl(std::make_tuple(n,
                                                        amgcl::make_iterator_range(ptr, ptr + n + 1),
                                                        amgcl::make_iterator_range(col, col + ptr[n]),
                                                        amgcl::make_iterator_range(val, val + ptr[n])));

    typename T::params prm;
    prm.solver.tol = opt.tolerance;
    prm.solver.maxiter = opt.iterations;
    prof.tic("setup_" + name);
    T solve(A_amgcl, prm);
    prof.toc("setup_" + name);
    returnvalue result;
    prof.tic("solve_" + name);
    std::tie(result.iterations, result.error) = solve(A_amgcl, b, x0);
    prof.toc("solve_" + name);
    std::cout << solve << std::endl;
    result.error_exact=(Axb.A*result_eigen-Axb.b).norm()/Axb.b.norm();
    return result;
}


template <class T>
returnvalue RunAMGCLCUDA_backend(amgcl::profiler<> &prof, Convoptions opt, const LinearSystem &Axb, const std::string &name)
{

	thrust::device_vector<double> X = std::vector<double>(Axb.x0.data(), Axb.x0.data() + Axb.x0.size());
	thrust::device_vector<double> F = std::vector<double>(Axb.b.data(), Axb.b.data() + Axb.b.size());
size_t n = Axb.A.rows();


    amgcl::backend::cuda<double>::params bprm;
    cusparseCreate(&bprm.cusparse_handle);
    typename T::params prm;
    prm.solver.tol = opt.tolerance;
    prm.solver.maxiter = opt.iterations;
    prof.tic("setup_" + name);
    T solve(Axb.A, prm,bprm);
    prof.toc("setup_" + name);
    returnvalue result;
    prof.tic("solve_" + name);
    std::tie(result.iterations, result.error) = solve(F, X);
    prof.toc("solve_" + name);
    std::cout << solve << std::endl;
    result.error_exact=(Axb.A*result_eigen-Axb.b).norm()/Axb.b.norm();
    return result;
}


int main(int argc, char *argv[])
{

    cxxopts::Options options("SparseSolverBench", "Benchmarking various sparse linear solvers");

    options.add_options()("t,threads", "How many threads to use", cxxopts::value<int>()->default_value("1"))("s,tolerance", "Tolerance to which solvers should converge", cxxopts::value<double>()->default_value("1e-12"))("i,iterations", "max number of iterations", cxxopts::value<int>()->default_value("1000"))("A,SparseMatrix", "MM Format File for Sparse Matrix", cxxopts::value<std::string>())("b,Rightside", "MM Format File for Vector", cxxopts::value<std::string>())("x,initialguess", "MM Format File for InitialGuess", cxxopts::value<std::string>())("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    std::string intial_guess_filename = "";
    if (result.count("initialguess"))
    {
        intial_guess_filename = result["initialguess"].as<std::string>();
    }

    Convoptions opt;

    int threads = result["threads"].as<int>();
    opt.tolerance = result["tolerance"].as<double>();
    opt.iterations = result["iterations"].as<int>();
    std::string A_filename = result["SparseMatrix"].as<std::string>();
    std::string b_filename = result["Rightside"].as<std::string>();

    std::cout << "Running profiler with " << threads << " threads\n"
              << "Required tolerance " << opt.tolerance << " num iterations " << opt.iterations << "\nA:" << A_filename << "\nb" << b_filename << std::endl;
    if (intial_guess_filename.size())
    {
        std::cout << "x0:" << intial_guess_filename << std::endl;
    }
    omp_set_num_threads(threads);

    amgcl::profiler<> prof;
    LinearSystem Axb;

    prof.tic("read_input");
    Eigen::loadMarket(Axb.A, A_filename);

    Eigen::loadMarketVector(Axb.b, b_filename);
    Axb.x0 = Eigen::VectorXd::Zero(Axb.A.rows());
    if (intial_guess_filename.size())
    {
        Eigen::loadMarketVector(Axb.x0, intial_guess_filename);
    }
    prof.toc("read_input");

    std::vector<std::string> names;
    std::vector<returnvalue> results;

    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::bicgstab<amgcl::backend::builtin<double>>>
        Solver_bicgstab;
names.push_back("amgcl_bicgstab");
  results.push_back(RunAMGCL_backend<Solver_bicgstab>(prof, opt, Axb, names.back()));

    typedef amgcl::make_solver<
        amgcl::relaxation::as_preconditioner<
        amgcl::backend::builtin<double>,
        amgcl::relaxation::ilu0>,
        amgcl::solver::bicgstab<amgcl::backend::builtin<double>>>
        Solver_bicgstab_ilut;
  names.push_back("amgcl_bicgstab_ilut");
  results.push_back(RunAMGCL_backend<Solver_bicgstab_ilut>(prof, opt, Axb, names.back()));

    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::bicgstabl<amgcl::backend::builtin<double>>>
        Solver_bicgstabl;
   names.push_back("amgcl_bicgstabl");
   results.push_back(RunAMGCL_backend<Solver_bicgstabl>(prof, opt, Axb, names.back()));

    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::idrs<amgcl::backend::builtin<double>>>
        Solver_idrs;

   names.push_back("amgcl_idrs");
   results.push_back(RunAMGCL_backend<Solver_idrs>(prof, opt, Axb, names.back()));

    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::eigen<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::bicgstab<amgcl::backend::eigen<double>>>
        Solver2;

   names.push_back("amgcl_bicgstab_eigen");
   results.push_back(RunAMGCLEigen_backend<Solver2>(prof, opt, Axb, names.back()));

  names.push_back("eigen_bicgstab");
results.push_back(RunEigen_Solver<Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>>>(prof, opt, Axb, names.back()));

    names.push_back("eigen_bicgstabl");
    results.push_back(RunEigen_Solver<Eigen::BiCGSTABL<Eigen::SparseMatrix<double, Eigen::RowMajor>>>(prof, opt, Axb, names.back()));

    names.push_back("eigen_idrstab");
    results.push_back(RunEigen_Solver<Eigen::IDRStab<Eigen::SparseMatrix<double, Eigen::RowMajor>>>(prof, opt, Axb, names.back()));



    typedef amgcl::make_solver<
        amgcl::amg<
        amgcl::backend::cuda<double>,
       amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::bicgstab<amgcl::backend::cuda<double>>>
        Solver_cuda_bicgstab;
    names.push_back("amgcl_cuda_bicgstab");
    results.push_back(RunAMGCLCUDA_backend<Solver_cuda_bicgstab>(prof, opt, Axb, names.back()));

    std::cout << "Eigen uses " << Eigen::nbThreads() << " threads" << std::endl;
    for(int i=0;i<names.size();i++){
        std::cout << names[i]<<"\t iter:" << results[i].iterations << "\t error " << results[i].error << "\t error_exact " << results[i].error_exact <<std::endl;
    }

    std::cout << prof << std::endl;
}
