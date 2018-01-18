#!/bin/bash

@CMAKE_BINARY_DIR@/src/geneo4PETSc --help # Help for coverage.
echo "" # Add space for clarity.

for f in "identity" "tridiag"
do
  for p in "-igs_pc_type#bjacobi"
  do
    l="log"
    if [[ "${p}" == *"identity"*   ]]; then l="mat"; fi
    if [[ "${p}" == *"tridiag"* ]]; then l="bin"; fi

    for m in "--metisDual" "--metisNodal"
    do
      PC_CMD="${p//[#]/ }"
      PC_LOG="${p//[#]/}"; PC_LOG="${PC_LOG//-/}"; PC_LOG="${PC_LOG//,/}"
      PC_LOG="${PC_LOG//igs_pc_type/}"

      OPT_LOG="${p//[#]/}"; OPT_LOG="${OPT_LOG//-/}"; OPT_LOG="${OPT_LOG//,/}"
      OPT_LOG="${OPT_LOG//igs_pc_type/}"
      OPT_LOG="${OPT_LOG//bjacobi/}"

      M_LOG="${m//--metisDual/dual}"; M_LOG="${M_LOG//--metisNodal/nodal}"

      LOG="${f}-pc=${PC_LOG}-metis=${M_LOG}"

      CMD="mpirun @MPIEXEC_PREFLAGS@ @MPIEXEC_NUMPROC_FLAG@ 2 @MPIEXEC_POSTFLAGS@" # Always use 2 processus to make sure log == ref.
      CMD="${CMD} @CMAKE_BINARY_DIR@/src/geneo4PETSc --inpFileA ${f}.inp"
      if [[ "${f}" == "identity" ]]; then CMD="${CMD} --inpFileB B.inp"; fi
      if [[ "${f}" == "tridiag" ]]; then CMD="${CMD} --inpEps 1."; fi
      CMD="${CMD} ${PC_CMD} --verbose 2 -geneo_chk log -geneo_dbg ${l},2 --shortRes"
      CMD="${CMD} -igs_ksp_atol 1.e-12 -igs_ksp_rtol 1.e-12" # Use tolerance to make "make test" as stable as possible.
      CMD="${CMD} -options_left no" # Get rid of unused option warnings with options_left (get clean logs).
      CMD="${CMD} ${m}"
      echo "$CMD" # Add command in the log: convienient to relaunch manually when problem occurs.
      CMD="${CMD} > ${LOG}.log 2>&1"
      eval "${CMD}"
      RC="$?"
      if [ "$RC" -ne "0" ]; then echo "ERROR"; exit 1; fi
      echo "" # Add space for clarity.

      @DIFF_EXECUTABLE@ "${LOG}.ref" "${LOG}.log"
      RC="$?"
      if [ "$RC" -ne "0" ]; then
        #vimdiff "${LOG}.ref" "${LOG}.log"
        exit 1
      fi
    done
  done
done

echo "OK"
