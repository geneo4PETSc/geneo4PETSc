# PETSc and SLEPc provide only a pc file (but no cmake file): to be consistent, geneo4PETSc provides only a pc file.
# MPI, Boost, PETSc and SLEPc must be resolved to link with geneo4PETSc: this is why all informations retrieved from
# MPI, Boost, PETSc.pc and SLEPc.pc are added to geneo4PETSc.pc.

set(GENEO4PETSC_DEP_HDR "")
foreach(hdr ${SLEPc_INCLUDE_DIRS} ${PETSc_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
  set(GENEO4PETSC_DEP_HDR "${GENEO4PETSC_DEP_HDR} -I${hdr}")
endforeach(hdr)

set(GENEO4PETSC_DEP_LIB "")
foreach(lib ${SLEPc_LDFLAGS} ${PETSc_LDFLAGS} ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})
  set(GENEO4PETSC_DEP_LIB "${GENEO4PETSC_DEP_LIB} ${lib}")
endforeach(lib)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/geneo4PETSc.pc ${CMAKE_CURRENT_BINARY_DIR}/geneo4PETSc.pc)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/geneo4PETSc.pc DESTINATION lib/pkgconfig)
