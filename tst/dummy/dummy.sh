#!/bin/bash

@CMAKE_BINARY_DIR@/src/geneo4PETSc --help # Help for coverage.
echo "" # Add space for clarity.

for f in "identity" "tridiag"
do
  for p in "-pc_type#bjacobi"                                  \
           "-pc_type#geneo#-geneo_lvl#ASM,0"                   \
           "-pc_type#geneo#-geneo_lvl#ASM,1"                   \
           "-pc_type#geneo#-geneo_lvl#ASM,1##--addOverlap#1"   \
           "-pc_type#geneo#-geneo_lvl#ASM,1##-geneo_offload"   \
           "-pc_type#geneo#-geneo_lvl#ASM,H1"                  \
           "-pc_type#geneo#-geneo_lvl#ASM,H1#--addOverlap#1"   \
           "-pc_type#geneo#-geneo_lvl#ASM,H1#-geneo_offload"   \
           "-pc_type#geneo#-geneo_lvl#ASM,E1"                  \
           "-pc_type#geneo#-geneo_lvl#ASM,E1#--addOverlap#1"   \
           "-pc_type#geneo#-geneo_lvl#ASM,E1#-geneo_offload"   \
           "-pc_type#geneo#-geneo_lvl#SORAS,0"                 \
           "-pc_type#geneo#-geneo_lvl#SORAS,2"                 \
           "-pc_type#geneo#-geneo_lvl#SORAS,2##--addOverlap#1" \
           "-pc_type#geneo#-geneo_lvl#SORAS,2##-geneo_offload" \
           "-pc_type#geneo#-geneo_lvl#SORAS,H2"                \
           "-pc_type#geneo#-geneo_lvl#SORAS,H2#--addOverlap#1" \
           "-pc_type#geneo#-geneo_lvl#SORAS,H2#-geneo_offload" \
           "-pc_type#geneo#-geneo_lvl#SORAS,E2"                \
           "-pc_type#geneo#-geneo_lvl#SORAS,E2#--addOverlap#1" \
           "-pc_type#geneo#-geneo_lvl#SORAS,E2#-geneo_offload"
  do
    l="log"
    if [[ "${p}" == *"identity"*   ]]; then l="mat"; fi
    if [[ "${p}" == *"tridiag"* ]]; then l="bin"; fi

    for m in "--metisDual" "--metisNodal"
    do
      PC_CMD="${p//[#]/ }"
      PC_LOG="${p//[#]/}"; PC_LOG="${PC_LOG//-/}"; PC_LOG="${PC_LOG//,/}"
      PC_LOG="${PC_LOG//pc_type/}"; PC_LOG="${PC_LOG//addOverlap1/}"
      PC_LOG="${PC_LOG//geneo_lvl/}"; PC_LOG="${PC_LOG//geneo_offload/}"

      OPT_LOG="${p//[#]/}"; OPT_LOG="${OPT_LOG//-/}"; OPT_LOG="${OPT_LOG//,/}"
      OPT_LOG="${OPT_LOG//geneo_offload/offload}"; OPT_LOG="${OPT_LOG//addOverlap/overlap}"
      OPT_LOG="${OPT_LOG//pc_type/}"
      OPT_LOG="${OPT_LOG//bjacobi/}"
      OPT_LOG="${OPT_LOG//geneo_lvl/}"; OPT_LOG="${OPT_LOG//geneo/}"
      OPT_LOG="${OPT_LOG//ASM0/}"; OPT_LOG="${OPT_LOG//ASM1/}"; OPT_LOG="${OPT_LOG//ASMH1/}"; OPT_LOG="${OPT_LOG//ASME1/}"
      OPT_LOG="${OPT_LOG//SORAS0/}"; OPT_LOG="${OPT_LOG//SORAS2/}"; OPT_LOG="${OPT_LOG//SORASH2/}"; OPT_LOG="${OPT_LOG//SORASE2/}"

      M_LOG="${m//--metisDual/dual}"; M_LOG="${M_LOG//--metisNodal/nodal}"

      LOG="${f}-pc=${PC_LOG}-metis=${M_LOG}"
      if [[ ! -z "${OPT_LOG}" ]]; then LOG="${LOG}-opt=${OPT_LOG}"; fi

      CMD="mpirun @MPIEXEC_PREFLAGS@ @MPIEXEC_NUMPROC_FLAG@ 2 @MPIEXEC_POSTFLAGS@" # Always use 2 processus to make sure log == ref.
      CMD="${CMD} @CMAKE_BINARY_DIR@/src/geneo4PETSc --inpFileA ${f}.inp"
      if [[ "${f}" == "identity" ]]; then CMD="${CMD} --inpFileB B.inp"; fi
      if [[ "${f}" == "tridiag" ]]; then CMD="${CMD} --inpEps 1. -geneo_cut 10"; fi # Add (useless !) -geneo_cut for coverage.
      CMD="${CMD} ${PC_CMD} --debug ${l} --verbose 2 -geneo_chk log -geneo_dbg ${l},2 --shortRes"
      CMD="${CMD} -ksp_atol 1.e-12 -ksp_rtol 1.e-12" # Use tolerance to make "make test" as stable as possible.
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
