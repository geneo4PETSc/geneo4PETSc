# Add test.

add_library(gengraph SHARED graph.cpp)
set_target_properties(gengraph PROPERTIES CXX_STANDARD 11)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/graphTest.sh"  "${CMAKE_CURRENT_BINARY_DIR}/graphTest.sh"  @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/graphRun.sh"   "${CMAKE_CURRENT_BINARY_DIR}/graphRun.sh"   @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/graphPlot.sh"  "${CMAKE_CURRENT_BINARY_DIR}/graphPlot.sh"  @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/graphPlot.ref" "${CMAKE_CURRENT_BINARY_DIR}/graphPlot.ref" @ONLY)
add_test(NAME graph COMMAND ./graphTest.sh WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
