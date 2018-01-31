# Add test.

add_library(genlaplacian SHARED laplacianServices.cpp laplacian.cpp)
target_include_directories(genlaplacian PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>) # Get laplacianServices.hpp
set_target_properties(genlaplacian PROPERTIES CXX_STANDARD 11)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/laplacianTest.sh"  "${CMAKE_CURRENT_BINARY_DIR}/laplacianTest.sh"  @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/laplacianRun.sh"   "${CMAKE_CURRENT_BINARY_DIR}/laplacianRun.sh"   @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/laplacianPlot.sh"  "${CMAKE_CURRENT_BINARY_DIR}/laplacianPlot.sh"  @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/laplacianPlot.ref" "${CMAKE_CURRENT_BINARY_DIR}/laplacianPlot.ref" @ONLY)
add_test(NAME laplacian COMMAND ./laplacianTest.sh WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
