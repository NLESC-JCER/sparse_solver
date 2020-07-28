foreach(PROG benchmark)
    file(GLOB ${PROG}_SOURCES ${PROG}*.cpp)
    add_executable(${PROG} ${${PROG}_SOURCES})
    target_link_libraries(${PROG} OpenMP::OpenMP_CXX Eigen3::Eigen amgcl::amgcl)
endforeach()

