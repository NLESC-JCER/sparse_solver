#
# Try to find ViennaCL library and include path.
# Once done this will define
#
# VIENNACL_FOUND
# VIENNACL_INCLUDE_DIRS
# VIENNACL_LIBRARIES
# VIENNACL_WITH_OPENCL
# 


option(VIENNACL_WITH_OPENCL "Use ViennaCL with OpenCL" FALSE)

IF(VIENNACL_WITH_OPENCL)
  find_package(OpenCL REQUIRED)
ENDIF(VIENNACL_WITH_OPENCL)


  find_path(VIENNACL_INCLUDE_DIR viennacl/forwards.h
    PATHS /usr/local/include
    DOC "The ViennaCL include path")

include(FindPackageHandleStandardArgs)
if(VIENNACL_WITH_OPENCL)
  set(VIENNACL_INCLUDE_DIRS ${VIENNACL_INCLUDE_DIR} ${OPENCL_INCLUDE_DIRS})
  set(VIENNACL_LIBRARIES ${OPENCL_LIBRARIES})
  find_package_handle_standard_args(ViennaCL "ViennaCL not found!" VIENNACL_INCLUDE_DIR OPENCL_INCLUDE_DIRS OPENCL_LIBRARIES)
else(VIENNACL_WITH_OPENCL)
  set(VIENNACL_INCLUDE_DIRS ${VIENNACL_INCLUDE_DIR})
  set(VIENNACL_LIBRARIES "")
  find_package_handle_standard_args(ViennaCL "ViennaCL not found!" VIENNACL_INCLUDE_DIR)
endif(VIENNACL_WITH_OPENCL)


MARK_AS_ADVANCED( VIENNACL_INCLUDE_DIR )