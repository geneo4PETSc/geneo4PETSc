#!/bin/bash -eu

# To run this, you need to configure PETSc this way:
# ~> ./configure --with-mpi=1 --with-pthread=1 --download-f2cblaslapack=yes              \
#                --download-mumps=yes --download-scalapack=yes                           \
#                --download-pastix=yes --download-ptscotch=yes                           \
#                --download-superlu=yes                                                  \
#                --download-superlu_dist=yes --download-parmetis=yes --download-metis=yes
#
# To run this, you need to configure SLEPc this way:
# ~> ./configure --download-arpack

# Note: you may want to have different sizes for strong / weak scaling.
#       for strong scaling, you may want to start with a big size to get a big enough size in the end.
#       for weak scaling, you may want to start with a smaller size to get a not too big (huge) size in the end.

# Note: GENEO_L2_ELS_OPT and GENEO_L2_ELS_TOL can increase dramatically the time needed to set up ASM or SORAS.
#       we need only to get "good enough" eigen vectors approximations.

STRONG_INP_SIZE="10"
STRONG_MPI="01 02"
WEAK_INP_SIZE="5"
WEAK_MPI="${STRONG_MPI}"
METIS=("--metisDual" "--metisNodal")
TOL="1.e-04 1.e-05"
PC="-pc_type#bjacobi -pc_type#mg"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,0"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,1"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,1##--addOverlap#1"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,1##-geneo_offload"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,H1"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,H1#--addOverlap#1"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,H1#-geneo_offload"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,E1"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,E1#--addOverlap#1"
PC="${PC} -pc_type#geneo#-geneo_lvl#ASM,E1#-geneo_offload"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,0"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,2"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,2##--addOverlap#1"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,2##-geneo_offload"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,H2"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,H2#--addOverlap#1"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,H2#-geneo_offload"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,E2"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,E2#--addOverlap#1"
PC="${PC} -pc_type#geneo#-geneo_lvl#SORAS,E2#-geneo_offload"
KSP_OPT=("-ksp_type#gmres") # "-ksp_type#cg")
OPT="-ksp_max_it 1000 -ksp_gmres_restart 1000" # Forbid GMRES restart (by imposing restart = max its).
OPT="${OPT} -options_left no" # Get rid of unused option warnings with options_left (get clean logs).
GENEO_L1_DLS=("-dls1_pc_factor_mat_solver_package#mumps") # "-dls1_pc_factor_mat_solver_package#superlu")
GENEO_L1_OPTIM=("-geneo_optim#0.00" "-geneo_optim#0.02")
GENEO_L2_TAU_GAMMA=("-geneo_tau#0.1#-geneo_gamma#8." "-geneo_tau#0.2#-geneo_gamma#12.")
GENEO_L2_ELS=("-els2_eps_type#arpack") # "-els2_eps_type#krylovschur#-geneo_no_syl")
GENEO_L2_ELS_OPT="-els2_eps_max_it 50" # Caution: the more iterations are performed, the more the eigen solve is costly.
GENEO_L2_ELS_TOL="-els2_eps_tol 1.e-02" # Caution: the most precise the tolerance is, the more the eigen solve is costly.
GENEO_OPT="" # "-geneo_cut 10"
MUMPS_OPT="-mat_mumps_cntl_1 0.01 -mat_mumps_cntl_3 -0.00001 -mat_mumps_cntl_4 0.00001" # Pivoting thresholds (mandatory for physical cases).
MG_OPT="-pc_mg_cycle_type w -pc_mg_smoothup 5 -pc_mg_smoothdown 5"
SUPERLU_OPT="-mat_superlu_replacetinypivot -mat_superlu_equil -mat_superlu_rowperm LargeDiag -mat_superlu_colperm NATURAL"
SUPERLUDIST_OPT="-mat_superlu_dist_replacetinypivot -mat_superlu_dist_equil -mat_superlu_dist_rowperm LargeDiag -mat_superlu_dist_colperm NATURAL"
VERBOSE="NO"
STOPONERROR="YES"

if [ "$#" -ne 1 ]; then echo "ERROR: need argument strong or weak (scaling)."; exit 1; fi
if [ "$1" != "strong" ] && [ "$1" != "weak" ] ; then echo "ERROR: need argument strong or weak (scaling)."; exit 1; fi

if [ "$1" == "strong" ]; then
  INP_SIZE_ARRAY=(${STRONG_INP_SIZE})
  MPI_ARRAY=(${STRONG_MPI})
else
  INP_SIZE_ARRAY=(${WEAK_INP_SIZE})
  MPI_ARRAY=(${WEAK_MPI})
fi
TOL_ARRAY=(${TOL})
PC_ARRAY=(${PC})

# Run jobs.

export OMP_PROC_BIND=TRUE

for s in "${INP_SIZE_ARRAY[@]}"
do
  for m in "${METIS[@]}"
  do
    for n in "${MPI_ARRAY[@]}"
    do
      for t in "${TOL_ARRAY[@]}"
      do
        for p in "${PC_ARRAY[@]}"
        do
          for k in "${KSP_OPT[@]}"
          do
            OTHER_OPT="${OPT}"

            unset OPT_ARRAY
            if [[ "${p}" == *"bjacobi"* ]]; then
              OPT_ARRAY+=("")
            elif [[ "${p}" == *"mg"* ]]; then
              OPT_ARRAY+=("")
            elif [[ "${p}" == *"geneo"*"0"* ]]; then # GenEO-0.
              OPT_ARRAY+=("${GENEO_L1_DLS[@]}")
            elif [[ "${p}" == *"geneo"*"1"* ]] || [[ "${p}" == *"geneo"*"2"* ]]; then # GenEO-1, GenEO-2.
              for d in "${GENEO_L1_DLS[@]}"
              do
                L1_OPTIM=(""); if [[ "${p}" == *"ORAS"* ]]; then L1_OPTIM=("${GENEO_L1_OPTIM[@]}"); fi
                for o in "${L1_OPTIM[@]}"
                do
                  for tg in "${GENEO_L2_TAU_GAMMA[@]}"
                  do
                    if [[ "${p}" == *"geneo"*"1"* ]]; then # GenEO-1.
                      tg="${tg%"#-geneo_gamma"*}" # Remove gamma: not needed.
                    fi

                    for e in "${GENEO_L2_ELS[@]}"
                    do
                      OPT_ARRAY+=("${d}#${tg}#${e}#${o}")
                    done
                  done
                done
              done
              OTHER_OPT="${OTHER_OPT} ${GENEO_OPT} ${GENEO_L2_ELS_OPT}"
            fi

            for o in "${OPT_ARRAY[@]}"
            do
              if [ "$1" == "strong" ]; then
                w="${MPI_ARRAY[0]}" # Strong scaling.
              else
                w="${n}" # Weak scaling.
              fi

              INP="--inpLibA ./@CMAKE_SHARED_LIBRARY_PREFIX@genlaplacian@CMAKE_SHARED_LIBRARY_SUFFIX@"
              INP="${INP} --size#${s}#--weakScaling#${w}#--kappa#2.#lin#--debug#--inpEps#0.0001"

              EPS="-ksp_atol ${t} -ksp_rtol ${t}";
              if [[ "${p}" == *"geneo"*"1"* ]] || [[ "${p}" == *"geneo"*"2"* ]]; then # GenEO-1, GenEO-2.
                EPS="${EPS} ${GENEO_L2_ELS_TOL}";
              fi

              KSP_CMD="${k//[#]/ }"
              KSP_LOG="${k//[#]/}"; KSP_LOG="${KSP_LOG//-/}"
              KSP_LOG="${KSP_LOG//ksp_type/}"

              OPT_CMD="${o//[#]/ }"
              OPT_LOG="${o//[#]/:}"; OPT_LOG="${OPT_LOG//-/}"
              OPT_LOG="${OPT_LOG//pc_factor_mat_solver_package/pc_solver}"; OPT_LOG="${OPT_LOG//els2_eps_type/els2_type}"

              M_LOG="${m//--metisDual/dual}"; M_LOG="${M_LOG//--metisNodal/nodal}"

              PC_CMD="${p//[#]/ }"
              PC_LOG="${p//[#]/}"; PC_LOG="${PC_LOG//-/}"; PC_LOG="${PC_LOG//,/}"
              PC_LOG="${PC_LOG//pc_type/}"; PC_LOG="${PC_LOG//addOverlap/overlap}"
              PC_LOG="${PC_LOG//geneo_lvl/}"; PC_LOG="${PC_LOG//geneo_offload/offload}"

              unset CMD
              CMD="mpirun @MPIEXEC_NUMPROC_FLAG@ ${n} @MPIEXEC_PREFLAGS@ @CMAKE_BINARY_DIR@/src/geneo4PETSc @MPIEXEC_POSTFLAGS@ --cmdLine"
              CMD="${CMD} ${INP} ${m} ${EPS} ${KSP_CMD} ${PC_CMD} ${OPT_CMD} ${OTHER_OPT} --timing"
              unset LOG
              LOG="laplacian-size=${s}-ws=${w}-np=${n}-tol=${t}-metis=${M_LOG}-ksp=${KSP_LOG}-pc=${PC_LOG}"
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
