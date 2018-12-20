cmake_minimum_required(VERSION 3.7)
enable_language(CXX)

# Check includes.

include(CheckIncludeFileCXX)
macro(check_include_file_cxx_error include_file)
  check_include_file_cxx("${include_file}" ${include_file}_FOUND)
  if(NOT ${include_file}_FOUND)
    message(FATAL_ERROR "${include_file} is not found")
  endif()
endmacro(check_include_file_cxx_error)
check_include_file_cxx_error(iostream)
check_include_file_cxx_error(string)
check_include_file_cxx_error(fstream)
check_include_file_cxx_error(sstream)
check_include_file_cxx_error(vector)
check_include_file_cxx_error(set)
check_include_file_cxx_error(iomanip)
check_include_file_cxx_error(chrono)
check_include_file_cxx_error(algorithm)
check_include_file_cxx_error(iterator)
check_include_file_cxx_error(cmath)
check_include_file_cxx_error(limits)

find_path(DL_INCLUDE_DIRS NAMES dlfcn.h PATHS $ENV{DL_DIR} $ENV{CPATH} $ENV{C_INCLUDE_PATH} $ENV{CPLUS_INCLUDE_PATH} PATH_SUFFIXES include)
if(NOT DL_INCLUDE_DIRS)
  message(FATAL_ERROR "dlfcn.h is not found")
endif()

# Find packages.

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmk")
find_package(Metis 5.1.0 REQUIRED)

find_package(MPI REQUIRED)

find_package(Boost REQUIRED COMPONENTS mpi serialization)

find_package(PkgConfig REQUIRED) # Get pkg_check_modules.
pkg_check_modules(PETSc REQUIRED PETSc>=3.10.3)
pkg_check_modules(SLEPc REQUIRED SLEPc>=3.10.1)

find_package(PythonInterp REQUIRED)

# Find programs.

find_program(DIFF_EXECUTABLE diff) # For test suite.

# Adding RPATH (must be done before add_executable or add_library).

set(CMAKE_MACOSX_RPATH TRUE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
foreach(dir "${PETSc_LIBRARY_DIRS}" "${SLEPc_LIBRARY_DIRS}")
  set(CMAKE_INSTALL_RPATH "${dir}:${CMAKE_INSTALL_RPATH}")
endforeach()
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Add pc file.

add_subdirectory(pkg)

# Add source code.

add_subdirectory(src)

# Add tests.

include(CTest)
enable_testing()
add_subdirectory(tst)

# Get summary for clarity.

function(summary title include libraries)
  message("   -- ${title}:")
  message("      -- compiling with:")
  foreach(inc ${include})
    message("            ${inc}")
  endforeach()
  message("      -- linking with:")
  foreach(lib ${libraries})
    message("            ${lib}")
  endforeach()
endfunction(summary)

message("-- Configuration summary:")
summary("DL utilities" "${DL_INCLUDE_DIRS}" "${CMAKE_DL_LIBS}")
summary("Metis" "${Metis_INCLUDE_DIRS}" "${Metis_LIBRARIES}")
summary("MPI" "${MPI_CXX_INCLUDE_PATH}" "${MPI_CXX_LIBRARIES}")
summary("Boost" "${Boost_INCLUDE_DIRS}" "${Boost_LIBRARY_DIRS}")
summary("PETSc" "${PETSc_INCLUDE_DIRS}" "${PETSc_LIBRARY_DIRS}")
summary("SLEPc" "${SLEPc_INCLUDE_DIRS}" "${SLEPc_LIBRARY_DIRS}")
message("   -- python: ${PYTHON_EXECUTABLE}")
message("   -- diff: ${DIFF_EXECUTABLE}")
