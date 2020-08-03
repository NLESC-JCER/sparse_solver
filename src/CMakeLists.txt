foreach(PROG benchmark)
    file(GLOB ${PROG}_SOURCES ${PROG}*.cpp)
    add_executable(${PROG} ${${PROG}_SOURCES})
    target_link_libraries(${PROG} PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen amgcl::amgcl)
    if(CUDA_FOUND)
        target_link_libraries(${PROG} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    endif()
endforeach()

