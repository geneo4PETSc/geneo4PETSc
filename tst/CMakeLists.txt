# Add test suite.

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/plot.py" "${CMAKE_CURRENT_BINARY_DIR}/plot.py")
add_subdirectory(dummy)
add_subdirectory(laplacian)
add_subdirectory(graph)
add_subdirectory(heat)

add_custom_target(check  COMMAND ${CMAKE_CTEST_COMMAND}    DEPENDS genlaplacian gengraph genheat)
add_custom_target(checkv COMMAND ${CMAKE_CTEST_COMMAND} -V DEPENDS genlaplacian gengraph genheat)
