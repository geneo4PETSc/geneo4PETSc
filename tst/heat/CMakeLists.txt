# Add test.

add_library(genheat SHARED heat.cpp ${CMAKE_SOURCE_DIR}/tst/laplacian/laplacianServices.cpp)
target_include_directories(genheat PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tst/laplacian>) # Get laplacianServices.hpp
set_target_properties(genheat PROPERTIES CXX_STANDARD 11)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/heatTest.sh" "${CMAKE_CURRENT_BINARY_DIR}/heatTest.sh" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/heatRun.sh"  "${CMAKE_CURRENT_BINARY_DIR}/heatRun.sh"  @ONLY)
add_test(NAME heat COMMAND ./heatTest.sh WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
