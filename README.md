geneo4PETSc - implementation of the GenEO preconditioner with PETSc and SLEPc.

Prerequisites:
==============

1. geneo4PETSc
   - CMake
   - dlopen utilities
   - Metis
   - MPI
   - Boost: Boost.MPI, Boost.Serialization
   - PETSc: configure --with-mpi=1 --download-mumps --download-scalapack ...
   - SLEPc: configure --download-arpack ...
2. test suite
   - python: argparse, numpy, matplotlib (for test suite)
   - diff (for test suite)

Note: if not done by "module load", you need to export PKG_CONFIG_PATH/CMAKE_PREFIX_PATH (check the wiki).

Note: prq.sh can help to build prerequisites (caution: export variables when script ends).  
~/geneo4PETSc> ./prq.sh petsc slepc  

Build and test:
===============

~/geneo4PETSc> pip install argparse numpy matplotlib  
~/geneo4PETSc> module load cmake metis mpi boost petsc slepc  
~/geneo4PETSc> mkdir BUILD; cd BUILD  
~/geneo4PETSc/BUILD> cmake ..; make all checkv  

Note: you need to check that all prerequisites are consistent unless link may break (check the wiki).

Usage:
======

~/geneo4PETSc/BUILD> ./src/geneo4PETSc --help

Note: in case the solver diverges, check the wiki.

Examples:
=========

Look at examples provided by the test suite.
