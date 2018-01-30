Welcome to the geneo4PETSc wiki!

geneo4PETSc is an implementation of the GenEO preconditioner with PETSc and SLEPc.

Some references associated to the GenEO (Generalized Eigenproblems in the Overlap) preconditioner are listed below:
- "Abstract robust coarse spaces for systems of PDEs via generalized eigenproblems in the overlaps" - N. Spillane, V. Dolean, P. Hauret, F. Nataf, C. Pechstein, R. Scheichl.
- "An additive Schwarz method type theory for Lions’s algorithm and a symmetrized optimized restricted additive Schwarz method" - R. Haferssas, P. Jolivet, F. Nataf.
- "An introduction to Domain Decomposition Methods - Algorithms, Theory, and Parallel Implementation" 1st edition - V. Dolean, P. Jolivet, F. Nataf.

In case you have a problem, check the related FAQ.

Install FAQ
===========

Prerequisites are not found:
----------------------------

If not done by your bashrc or "module load", you need to export PKG_CONFIG_PATH and/or CMAKE_PREFIX_PATH and/or hints.

Typically, to find PETSc and SLEPc, you need to export:
- export PKG_CONFIG_PATH="/path/to/petsc/local/lib/pkgconfig:${PKG_CONFIG_PATH}"
- export PKG_CONFIG_PATH="/path/to/slepc/local/lib/pkgconfig:${PKG_CONFIG_PATH}"
- export CMAKE_PREFIX_PATH="/path/to/petsc/local:${CMAKE_PREFIX_PATH}"
- export CMAKE_PREFIX_PATH="/path/to/slepc/local:${CMAKE_PREFIX_PATH}"

Typically, if you do not use the Metis library compiled by PETSc, to give a hint to find Metis you need to export:
- export Metis_DIR="/path/to/metis/local"
- export CMAKE_PREFIX_PATH="/path/to/metis/local:${CMAKE_PREFIX_PATH}"

Minimal configuration of PETSc and SLEPc:
-----------------------------------------

- ~/petsc> ./configure --with-mpi=1 --download-mumps --download-scalapack ...
- ~/slepc> ./configure --download-arpack ...

Headers are not found:
----------------------

- STL C++ header: did you install a recent C++ compiler (C++11) and libstdc++-dev ?
- dlfcn.h: did you install libc-dev ?

Symbols are not found when linking:
-----------------------------------

You need to check that all prerequisites are consistent (unless link may break). Typically, the MPI/Boost/Metis installs found by CMake must be the ones used by PETSc and SLEPc.

In the build directory, remove CMakeCache.txt if any. Set environment variables (PKG_CONFIG_PATH, CMAKE_PREFIX_PATH) before running cmake. Check paths are correct in the cmake summary before the build.

Not working on Windows:
-----------------------

Use debian. Windows is not meant to be supported.

Usage FAQ
=========

How to run the test suite:
--------------------------

- make check: builds the test utilities and run the test suite.
- make checkv: does the same as "make check" but with verbosity.
- make test: runs the test suite but don't build the test utilities.

The test suite fails:
---------------------

Did you configure PETSc and SLEPc with the minimal following options:
- ~/petsc> ./configure --with-mpi=1 --download-mumps --download-scalapack ...
- ~/slepc> ./configure --download-arpack ...

The solver diverges:
--------------------

- GenEO is dedicated to SPD matrices: adding -geneo_chk can help to check this.
- Adding -geneo_dbg can help to understand the problem (create debug files globally and/or per domain).

Testing with "too" small matrices and "lots" of domains may fail:
-----------------------------------------------------------------

GenEO is based on local eigen value problems (one per domain). If the A matrix is "too small" and the number of domains is "too big", the local eigen value problems may not be relevant (no physical meaning): the solve may break.

Use "big enough" or "real life" (physical) matrices.
