# Create project and target(s).

project(geneo4PETSc)

add_executable(geneo4PETSc geneo4PETSc.cpp)
set_target_properties(geneo4PETSc PROPERTIES CXX_STANDARD 11)
install(TARGETS geneo4PETSc RUNTIME DESTINATION bin)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Before add_library, allow the archive to be included in a shared object.
add_library(geneopc STATIC geneo.cpp)
set_target_properties(geneopc PROPERTIES CXX_STANDARD 11)
target_include_directories(geneopc PRIVATE "../hdr")
install(TARGETS geneopc ARCHIVE DESTINATION lib)

# Set geneo dependencies.

target_include_directories(geneopc PRIVATE "${MPI_CXX_INCLUDE_PATH}")
target_link_libraries(geneopc PRIVATE "${MPI_CXX_LIBRARIES}")

target_include_directories(geneopc PRIVATE "${Boost_INCLUDE_DIRS}")
target_link_libraries(geneopc PRIVATE "${Boost_LIBRARIES}")

target_include_directories(geneopc PRIVATE "${PETSc_INCLUDE_DIRS}")
target_link_libraries(geneopc PRIVATE "${PETSc_LDFLAGS}")

target_include_directories(geneopc PRIVATE "${SLEPc_INCLUDE_DIRS}")
target_link_libraries(geneopc PRIVATE "${SLEPc_LDFLAGS}")

# Set geneo4PETSc dependencies.

target_include_directories(geneo4PETSc PRIVATE "${DL_INCLUDE_DIRS}")
target_link_libraries(geneo4PETSc PRIVATE "${CMAKE_DL_LIBS}") # Link with dlopen utilities.

target_include_directories(geneo4PETSc PRIVATE "${Metis_INCLUDE_DIRS}")
target_link_libraries(geneo4PETSc PRIVATE "${Metis_LIBRARIES}")

target_include_directories(geneo4PETSc PRIVATE "${MPI_CXX_INCLUDE_PATH}")
target_link_libraries(geneo4PETSc PRIVATE "${MPI_CXX_LIBRARIES}")

target_include_directories(geneo4PETSc PRIVATE "${Boost_INCLUDE_DIRS}")
target_link_libraries(geneo4PETSc PRIVATE "${Boost_LIBRARIES}")

target_include_directories(geneo4PETSc PRIVATE "${PETSc_INCLUDE_DIRS}")
target_link_libraries(geneo4PETSc PRIVATE "${PETSc_LDFLAGS}")

target_include_directories(geneo4PETSc PRIVATE "${SLEPc_INCLUDE_DIRS}")
target_link_libraries(geneo4PETSc PRIVATE "${SLEPc_LDFLAGS}")

target_include_directories(geneo4PETSc PRIVATE "../hdr")
target_link_libraries(geneo4PETSc PRIVATE geneopc)
