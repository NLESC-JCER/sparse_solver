foreach(PROG benchmark)
    file(GLOB ${PROG}_SOURCES ${PROG}*.cu)
    cuda_add_executable(${PROG} ${${PROG}_SOURCES})
    target_link_libraries(${PROG} OpenMP::OpenMP_CXX Eigen3::Eigen amgcl::amgcl)
    if(CUDA_FOUND)
        target_link_libraries(${PROG} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
    endif()
endforeach()

