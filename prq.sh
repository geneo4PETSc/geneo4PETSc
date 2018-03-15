#!/bin/bash -eu

function errorMsg {
  if [[ "$?" -ne "0" ]]; then echo "ERROR: check prq/*/prq.*.log"; fi
}
trap 'errorMsg' EXIT

export PRQ_DIR; PRQ_DIR="$(pwd)/prq"

echo "This script is meant to help installing prerequisites."
echo "Call this script with any of the following arguments: openmpi boost petsc slepc"
if [[ "$#" == "0" ]]; then exit 0; fi

echo ""
echo "This script is NOT meant to fix ALL problems one may have on ANY plateforms."
echo "Adapt it as needed depending on specific needs or problems you may encounter."

mkdir -p "${PRQ_DIR}"
cd "${PRQ_DIR}"

# Note: the ${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} construct is used to avoid error if variable is not defined yet.
export PATH="${PRQ_DIR}/local/bin:${PATH:+:$PATH}"
export LD_LIBRARY_PATH="${PRQ_DIR}/local/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${PRQ_DIR}/local/lib/pkgconfig:${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CMAKE_PREFIX_PATH="${PRQ_DIR}/local:${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

for OPT in "$@"
do
  echo ""
  echo "**** $OPT ****"
  echo ""

  if [[ "$OPT" == "openmpi" ]]; then
    if [[ -f "${PRQ_DIR}/local/bin/mpirun" ]]; then echo "done."; continue; fi

    if [[ ! -f openmpi-1.8.5.tar.gz ]]; then wget https://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.5.tar.gz; fi
    tar -xf openmpi-1.8.5.tar.gz
    pushd openmpi-1.8.5

    echo -e "\\nconfiguring..."
    ./configure --prefix="${PRQ_DIR}/local" --with-devel-headers --enable-mpi-cxx --enable-mpi-fortran &> prq.openmpi.configure.log
    tail -n 25 prq.openmpi.configure.log
    echo -e "\\nbuilding..."
    make -j                                                                                            &> prq.openmpi.make.log
    tail -n 25 prq.openmpi.make.log
    echo -e "\\ninstalling..."
    make install                                                                                       &> prq.openmpi.install.log
    tail -n 25 prq.openmpi.install.log
    popd
  elif [[ "$OPT" == "boost" ]]; then
    if [[ -f "${PRQ_DIR}/local/lib/libboost_mpi.a" ]]; then echo "done."; continue; fi

    if [[ ! -f boost_1_61_0.tar.gz ]]; then wget https://sourceforge.net/projects/boost/files/boost/1.61.0/boost_1_61_0.tar.gz; fi
    tar -xf boost_1_61_0.tar.gz
    pushd boost_1_61_0

    echo -e "\\nbootstraping..."
    ./bootstrap.sh --prefix="${PRQ_DIR}/local" --with-libraries=mpi,serialization &> prq.boost.bootstrap.log
    tail -n 25 prq.boost.bootstrap.log
    cp -f project-config.jam user-config.jam
    echo "using mpi : ${PRQ_DIR}/local/bin/mpic++ ;" >> user-config.jam
    echo -e "\\nbuilding..."
    ./b2 --user-config=user-config.jam                                            &> prq.boost.make.log
    tail -n 25 prq.boost.make.log
    echo -e "\\ninstalling..."
    ./b2 --user-config=user-config.jam install                                    &> prq.boost.install.log
    tail -n 25 prq.boost.install.log
    popd
  elif [[ "$OPT" == "petsc" ]]; then
    if [[ ! -d petsc/.git ]]; then rm -fr petsc; git clone https://bitbucket.org/petsc/petsc; fi
    pushd petsc
    git checkout master
    git pull origin master

    rm -f arch*/lib/petsc/conf/pkg.* # Remove to force reconfigure if petsc is cached.
    export MPI_FLAGS=""
    if [[ -f "${PRQ_DIR}/local/bin/mpirun" ]]; then
      export MPI_FLAGS="--with-mpi-dir=${PRQ_DIR}/local";
      echo -e "\\nexport MPI_FLAGS=${MPI_FLAGS}"
    fi

    echo -e "\\nconfiguring..."
    ./configure --prefix="${PRQ_DIR}/local" --PETSC_ARCH=arch \
                --with-mpi=1 "${MPI_FLAGS}" --with-fortran-bindings=0 \
                --download-metis=yes --download-mumps=yes --download-scalapack=yes &> prq.petsc.configure.log
    tail -n 25 prq.petsc.configure.log
    echo -e "\\nbuilding..."
    make PETSC_DIR="$PWD" PETSC_ARCH=arch     all                                  &> prq.petsc.make.log
    tail -n 25 prq.petsc.make.log
    echo -e "\\ninstalling..."
    make PETSC_DIR="$PWD" PETSC_ARCH=arch install                                  &> prq.petsc.install.log
    tail -n 25 prq.petsc.install.log
    popd
  elif [[ "$OPT" == "slepc" ]]; then
    if [[ ! -d slepc/.git ]]; then rm -fr slepc; git clone https://bitbucket.org/slepc/slepc; fi
    pushd slepc
    git checkout master
    git pull origin master

    export PETSC_DIR="${PRQ_DIR}/local"
    export SLEPC_DIR="$PWD"
    echo -e "\\nconfiguring..."
    ./configure --prefix="${PRQ_DIR}/local"  --download-arpack &> prq.slepc.configure.log
    tail -n 25 prq.slepc.configure.log
    echo -e "\\nbuilding..."
    make SLEPC_DIR="$SLEPC_DIR" PETSC_DIR="$PETSC_DIR"     all &> prq.slepc.make.log
    tail -n 25 prq.slepc.make.log
    echo -e "\\ninstalling..."
    make SLEPC_DIR="$SLEPC_DIR" PETSC_DIR="$PETSC_DIR" install &> prq.slepc.install.log
    tail -n 25 prq.slepc.install.log
    popd
  fi
done

echo ""
echo "Now you need to export:"
echo ""
echo "export PATH=\"${PRQ_DIR}/local/bin:\${PATH}\""
echo "export LD_LIBRARY_PATH=\"${PRQ_DIR}/local/lib:\${LD_LIBRARY_PATH}\""
echo "export PKG_CONFIG_PATH=\"${PRQ_DIR}/local/lib/pkgconfig:\${PKG_CONFIG_PATH}\""
echo "export CMAKE_PREFIX_PATH=\"${PRQ_DIR}/local:\${CMAKE_PREFIX_PATH}\""
