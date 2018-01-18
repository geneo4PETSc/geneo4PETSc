# Create project and target(s).

project(geneo4PETSc)

add_executable(geneo4PETSc geneo4PETSc.cpp)
set_target_properties(geneo4PETSc PROPERTIES CXX_STANDARD 11)
install(TARGETS geneo4PETSc RUNTIME DESTINATION bin)

# Set project dependencies.

target_include_directories(geneo4PETSc PRIVATE "${DL_INCLUDE_DIRS}")
target_link_libraries(geneo4PETSc PRIVATE "${CMAKE_DL_LIBS}") # Link with dlopen utilities.

target_include_directories(geneo4PETSc PRIVATE "${Metis_INCLUDE_DIRS}")
target_link_libraries(geneo4PETSc PRIVATE "${Metis_LIBRARIES}")

target_include_directories(geneo4PETSc PRIVATE "${MPI_CXX_INCLUDE_PATH}")
target_link_libraries(geneo4PETSc PRIVATE "${MPI_CXX_LIBRARIES}")

target_include_directories(geneo4PETSc PRIVATE "${Boost_INCLUDE_DIRS}")
target_link_libraries(geneo4PETSc PRIVATE "${Boost_LIBRARIES}")

target_include_directories(geneo4PETSc PRIVATE "${PETSc_INCLUDE_DIRS}")
foreach(lib "${PETSc_LDFLAGS}")
  target_link_libraries(geneo4PETSc PRIVATE "${lib}")
endforeach()
