# sparse_solver

This is a small benchmark for linear solvers from the Eigen and amgcl library.

This version requires the linear solver MR from Eigen

## Installation

Download and then run

`mkdir build`
`cd build`
`cmake -DCMAKE_PREFIX_PATH="PATH_TO_AMGCL;PATH_TO_EIGEN" ..`
`make`

In `build/src` you will find a `benchmark` executable with `-h` you get help options. 

For running it requires a sparse matrix `A` and a vector `b` in the MatrixMarket format. 
