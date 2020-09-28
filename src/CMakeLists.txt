foreach(PROG benchmark)
    file(GLOB ${PROG}_SOURCES ${PROG}*.cpp)
    
    if(USE_CUDA)
    set_source_files_properties( ${PROG} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
        cuda_add_executable(${PROG} ${${PROG}_SOURCES} ${SPARSE_SOLVERS_HEADERS})
    else()
        add_executable(${PROG} ${${PROG}_SOURCES} ${SPARSE_SOLVERS_HEADERS})
    endif()

    target_include_directories(${PROG} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${PROG} OpenMP::OpenMP_CXX Eigen3::Eigen amgcl::amgcl)
    if(USE_CUDA)
        target_link_libraries(${PROG} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
    endif()
endforeach()

