#!/bin/bash

# To run this, you need to configure PETSc this way:
# ~> ./configure --with-mpi=1 --with-pthread=1 --download-f2cblaslapack=yes              \
#                --download-mumps=yes --download-scalapack=yes                           \
#                --download-pastix=yes --download-ptscotch=yes                           \
#                --download-superlu=yes                                                  \
#                --download-superlu_dist=yes --download-parmetis=yes --download-metis=yes

# Note: you may want to have different sizes for strong / weak scaling.
#       for strong scaling, you may want to start with a big size to get a big enough size in the end.
#       for weak scaling, you may want to start with a smaller size to get a not too big (huge) size in the end.

STRONG_INP_SIZE="10"
STRONG_INP_LEVEL="2"
STRONG_MPI="1 @MPIEXEC_MAX_NUMPROCS@"
WEAK_INP_SIZE="5"
WEAK_INP_LEVEL="2"
WEAK_MPI="${STRONG_MPI}"
METIS=("--metisDual" "--metisNodal")
TOL="1.e-04 1.e-05"
PC="-igs_pc_type#bjacobi -igs_pc_type#mg"
IGS_TYPE_OPT=("-igs_ksp_type#gmres") # "-igs_ksp_type#cg")
IGS_OPT="-igs_ksp_max_it 1000 -igs_ksp_gmres_restart 1000" # Forbid GMRES restart (by imposing restart = max its).
IGS_OPT="${IGS_OPT} -options_left no" # Get rid of unused option warnings with options_left (get clean logs).
MUMPS_OPT="-mat_mumps_cntl_1 0.01 -mat_mumps_cntl_3 -0.00001 -mat_mumps_cntl_4 0.00001" # Pivoting thresholds (mandatory for physical cases).
MG_OPT="-igs_pc_mg_cycle_type w -igs_pc_mg_smoothup 5 -igs_pc_mg_smoothdown 5"
SUPERLU_OPT="-mat_superlu_replacetinypivot -mat_superlu_equil -mat_superlu_rowperm LargeDiag -mat_superlu_colperm NATURAL"
SUPERLUDIST_OPT="-mat_superlu_dist_replacetinypivot -mat_superlu_dist_equil -mat_superlu_dist_rowperm LargeDiag -mat_superlu_dist_colperm NATURAL"
VERBOSE="NO"
STOPONERROR="YES"

if [ "$#" -ne 1 ]; then echo "ERROR: need argument strong or weak (scaling)."; exit 1; fi
if [ "$1" != "strong" ] && [ "$1" != "weak" ] ; then echo "ERROR: need argument strong or weak (scaling)."; exit 1; fi

if [ "$1" == "strong" ]; then
  INP_SIZE_ARRAY=(${STRONG_INP_SIZE})
  INP_LEVEL_ARRAY=(${STRONG_INP_LEVEL})
  MPI_ARRAY=(${STRONG_MPI})
else
  INP_SIZE_ARRAY=(${WEAK_INP_SIZE})
  INP_LEVEL_ARRAY=(${WEAK_INP_LEVEL})
  MPI_ARRAY=(${WEAK_MPI})
fi
TOL_ARRAY=(${TOL})
PC_ARRAY=(${PC})

# Run jobs.

export OMP_PROC_BIND=TRUE

for s in "${INP_SIZE_ARRAY[@]}"
do
  for l in "${INP_LEVEL_ARRAY[@]}"
  do
    for m in "${METIS[@]}"
    do
      for n in "${MPI_ARRAY[@]}"
      do
        for t in "${TOL_ARRAY[@]}"
        do
          for p in "${PC_ARRAY[@]}"
          do
            for k in "${IGS_TYPE_OPT[@]}"
            do
              OTHER_OPT="${IGS_OPT}"

              unset OPT_ARRAY
              if [[ "${p}" == *"bjacobi"* ]]; then
                OPT_ARRAY+=("")
              elif [[ "${p}" == *"mg"* ]]; then
                OPT_ARRAY+=("")
              fi

              for o in "${OPT_ARRAY[@]}"
              do
                if [ "$1" == "strong" ]; then
                  w="${MPI_ARRAY[0]}" # Strong scaling.
                else
                  w="${n}" # Weak scaling.
                fi

                INP="--inpLibA ./libgengraph@CMAKE_SHARED_LIBRARY_SUFFIX@ --size#${s}#--level#${l}#--weakScaling#${w}#--noGround"

                EPS="-igs_ksp_atol ${t} -igs_ksp_rtol ${t}";

                KSP_CMD="${k//[#]/ }"
                KSP_LOG="${k//[#]/}"; KSP_LOG="${KSP_LOG//-/}"
                KSP_LOG="${KSP_LOG//igs_ksp_type/}"

                OPT_CMD="${o//[#]/ }"
                OPT_LOG="${o//[#]/:}"; OPT_LOG="${OPT_LOG//-/}"
                OPT_LOG="${OPT_LOG//pc_factor_mat_solver_package/pc_solver}"; OPT_LOG="${OPT_LOG//els2_eps_type/els2_type}"

                M_LOG="${m//--metisDual/dual}"; M_LOG="${M_LOG//--metisNodal/nodal}"

                PC_CMD="${p//[#]/ }"
                PC_LOG="${p//[#]/}"; PC_LOG="${PC_LOG//-/}"; PC_LOG="${PC_LOG//,/}"
                PC_LOG="${PC_LOG//igs_pc_type/}"; PC_LOG="${PC_LOG//addOverlap/overlap}"

                unset CMD
                CMD="mpirun @MPIEXEC_NUMPROC_FLAG@ ${n} @MPIEXEC_PREFLAGS@ @CMAKE_BINARY_DIR@/src/geneo4PETSc @MPIEXEC_POSTFLAGS@ --cmdLine"
                CMD="${CMD} ${INP} ${m} ${EPS} ${KSP_CMD} ${PC_CMD} ${OPT_CMD} ${OTHER_OPT} --timing"
                unset LOG
                LOG="graph-size=${s}-level=${l}-ws=${w}-np=${n}-tol=${t}-metis=${M_LOG}-ksp=${KSP_LOG}-pc=${PC_LOG}"
                if [[ ! -z "${OPT_LOG}" ]]; then LOG="${LOG}-opt=${OPT_LOG}"; fi
                LOG="${LOG}.log"

                if [[ "${CMD}" == *"mumps"* ]]; then CMD="${CMD} ${MUMPS_OPT}"; fi
                if [[ "${CMD}" == *"mg"* ]]; then CMD="${CMD} ${MG_OPT}"; fi
                if [[ "${CMD}" == *"superlu "* ]]; then CMD="${CMD} ${SUPERLU_OPT}"; fi
                if [[ "${CMD}" == *"superlu_dist "* ]]; then CMD="${CMD} ${SUPERLUDIST_OPT}"; fi

                if [ ! -f "${LOG}" ]; then
                  echo "$CMD" # Add command in the log: convienient to relaunch manually when problem occurs.
                  eval "${CMD} > ${LOG} 2>&1"
                  RC="$?"
                  if [[ "$VERBOSE" == "YES" ]]; then more "$LOG"; fi
                  if [ "$RC" -ne "0" ]; then mv "$LOG" "$LOG.ko"; fi
                  if [[ "$STOPONERROR" == "YES" ]]; then
                    if [ "$RC" -ne "0" ]; then echo "ERROR"; more "$LOG.ko"; exit 1; fi
                  fi
                  echo ""
                fi
              done
            done
          done
        done
      done
    done
  done
done
