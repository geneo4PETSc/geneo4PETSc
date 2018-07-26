/*
 * This is an implementation of the GenEO preconditioner with PETSc and SLEPc.
 *
 * references:
 *   - R1: "An introduction to Domain Decomposition Methods - Algorithms, Theory, and Parallel Implementation" 1st edition
 *         V. Dolean, P. Jolivet, F. Nataf.
 *   - R2: "Abstract robust coarse spaces for systems of PDEs via generalized eigenproblems in the overlaps".
 *         N. Spillane, V. Dolean, P. Hauret, F. Nataf, C. Pechstein, R. Scheichl.
 *   - R3: "Deflation of conjugate gradients with applications to boundary value problems".
 *         R. Nicolaides.
 *   - R4: "An additive Schwarz method type theory for Lions’s algorithm and a symmetrized optimized restricted additive
 *          Schwarz method"
 *         R. Haferssas, P. Jolivet, F. Nataf.
 *
 * conventions:
 *   - all variables prefixed "pc"  are PETSc class/struct instances.
 *   - all variables prefixed "sc"  are SLEPc class/struct instances.
 *   - all variables suffixed "Loc" are associated to local matrices (domain).
 *     Note: conversely, all variables that are not suffixed "Loc" are associated to the global problem.
 *   - all variables suffixed "E2L" are associated to matrix going from eigen space to local  space.
 *   - all variables suffixed "E2G" are associated to matrix going from eigen space to global space.
 *   - all variables suffixed "Eig" are associated to matrix going from eigen space to eigen  space.
 *   - all variables suffixed "Off" are offloaded on master (local copy of a global data).
 *
 * notes:
 *   - A and Z: as they can be made from index sets (IS) that may overlap, they are built using MatIS.
 *   - when building a MatIS:
 *     - first, a local matrix is created (preallocated, filled) per domain (MPI proc).
 *     - then, local matrices are gathered by the (global/distributed) MatIS (MatISSetLocalMat).
 *   - some operations are NOT supported on MatIS (in particular PtAP): if needed, a MatMPI version
 *     (which supports PtAP) of a MatIS matrix may be created (MatISGetMPIXAIJ).
 */

#include "geneo.hpp"

#include <string>
#include <fstream> // ofstream.
#include <sstream> // stringstream.
#include <vector>
#include <set>
#include <iomanip> // setfill.
#include <chrono>
#include <cmath> // fabs.
#include <algorithm> // max_element, min_element.
#include <limits> // numeric_limits.

#include <petsc.h>
#include <petsc/private/pcimpl.h> // Private include file intended for use by all preconditioner.
#include <slepc.h>

#include <boost/mpi.hpp>

using namespace std;

PetscErrorCode createViewer(bool const & bin, bool const & mat, MPI_Comm const & comm, string const & baseName,
                            PetscViewer & pcView) {
  if (bin) { // Binary.
    string fileName = baseName + ".bin";
    return PetscViewerBinaryOpen(comm, fileName.c_str(), FILE_MODE_WRITE, &pcView);
  }
  else { // ASCII.
    string fileName = baseName;
    fileName += (mat) ? ".mat" : ".log";
    PetscErrorCode pcRC = PetscViewerASCIIOpen(comm, fileName.c_str(), &pcView);
    CHKERRQ(pcRC);
    if (mat) {
      pcRC = PetscViewerPushFormat(pcView, PETSC_VIEWER_ASCII_MATLAB);
      CHKERRQ(pcRC);
    }
    return pcRC;
  }
}

#define SETERRABT(msg) SETERRABORT(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,msg)

PetscErrorCode tuneSolver(PC const & pcPC, Mat const & pcFactor, bool inertia = false) {
  MatSolverType pcType = NULL;
  PetscErrorCode pcRC = PCFactorGetMatSolverType(pcPC, &pcType);
  CHKERRQ(pcRC);
  if (pcType && string(pcType) == "mumps") {
    pcRC = MatMumpsSetIcntl(pcFactor, 24, 1); // Detection of null pivots (avoided if possible).
    CHKERRQ(pcRC);
    pcRC = MatMumpsSetCntl(pcFactor, 5, 1.e+20); // Fixing null pivots (when not possible to avoid them).
    CHKERRQ(pcRC);
    if (inertia) {
      pcRC = MatMumpsSetIcntl(pcFactor, 13, 1); // To get relevant inertia.
      CHKERRQ(pcRC);
    }
  }

  return 0;
}

PetscErrorCode directLocalSolve(geneoContext const * const gCtx, KSP const & pcKSPLoc) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Set up a direct solver.

  PetscErrorCode pcRC = KSPSetType(pcKSPLoc, KSPPREONLY); // Direct solve: apply only preconditioner (not A).
  CHKERRQ(pcRC);
  PC pcPCLoc;
  pcRC = KSPGetPC(pcKSPLoc, &pcPCLoc);
  CHKERRQ(pcRC);
  pcRC = PCSetType(pcPCLoc, PCLU); // Direct solve: preconditioner "of type LU" <=> direct solver.
  CHKERRQ(pcRC);
  pcRC = PCFactorSetMatSolverType(pcPCLoc, "mumps");
  CHKERRQ(pcRC);
  pcRC = KSPSetFromOptions(pcKSPLoc); // Just before setup to override default options.
  CHKERRQ(pcRC);
  pcRC = KSPSetUp(pcKSPLoc);
  CHKERRQ(pcRC);
  PCType pcPCTypeLoc;
  pcRC = PCGetType(pcPCLoc, &pcPCTypeLoc); // May has been changed by the command line.
  CHKERRQ(pcRC);
  if (pcPCTypeLoc && string(pcPCTypeLoc) == "lu") {
    Mat pcFactorLoc;
    pcRC = PCFactorGetMatrix(pcPCLoc, &pcFactorLoc); // Force LU factorisation (stand by for later use).
    CHKERRQ(pcRC);
    pcRC = tuneSolver(pcPCLoc, pcFactorLoc);
    CHKERRQ(pcRC);
  }

  return 0;
}

PetscErrorCode setUpLevel1(geneoContext * const gCtx, Mat const * const pcADirLoc, Mat const & pcARobLoc) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");
  if (!pcADirLoc) SETERRABT("GenEO preconditioner without dirichlet matrix");

  // Create solver for local dirichlet problem (LU factorisation).

  auto start = chrono::high_resolution_clock::now();
  PetscErrorCode pcRC = KSPCreate(PETSC_COMM_SELF, &(gCtx->pcKSPL1Loc)); // Local solve (delegated to each MPI proc).
  CHKERRQ(pcRC);
  pcRC = KSPSetOptionsPrefix(gCtx->pcKSPL1Loc, "dls1_");
  CHKERRQ(pcRC);
  if (gCtx->lvl1ORAS) { // (14) from R4.
    pcRC = KSPSetOperators(gCtx->pcKSPL1Loc, pcARobLoc, pcARobLoc); // Set Robin matrix as operator.
    CHKERRQ(pcRC);
  }
  else { // (1.30) from R1.
    pcRC = KSPSetOperators(gCtx->pcKSPL1Loc, *pcADirLoc, *pcADirLoc); // Set Dirichlet matrix as operator.
    CHKERRQ(pcRC);
  }
  pcRC = directLocalSolve(gCtx, gCtx->pcKSPL1Loc);
  CHKERRQ(pcRC);
  auto stop = chrono::high_resolution_clock::now();
  gCtx->lvl1SetupMinvTimeLoc = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  // Create local vector and scatter-gather context.

  pcRC = MatCreateVecs(gCtx->pcA, &(gCtx->pcX), NULL); // Vector matching the matrix (avoid handling layout).
  CHKERRQ(pcRC);
  pcRC = VecCreateSeq(PETSC_COMM_SELF, gCtx->nbDOFLoc, &(gCtx->pcXLoc));
  CHKERRQ(pcRC);
  pcRC = VecScatterCreate(gCtx->pcX, gCtx->pcIS, gCtx->pcXLoc, NULL, &(gCtx->pcScatCtx));
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode preallocateZE2L(PetscInt const & pcNbEVLoc, geneoContext const * const gCtx, Mat & pcZE2L) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Preallocate the local part of coarse space.

  PetscErrorCode pcRC = MatCreateSeqDense(PETSC_COMM_SELF, gCtx->nbDOFLoc, pcNbEVLoc, NULL, &pcZE2L);
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode checkRank(bool const & localSolve, Mat const & pcM, PetscInt const & pcNbEV, geneoContext const * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  if (pcNbEV == 0) return 0; // Nothing to do.

  // Create a basis vector.

  BV scBV;
  PetscErrorCode pcRC;
  if (localSolve) pcRC = BVCreate(PETSC_COMM_SELF,  &scBV);
  else            pcRC = BVCreate(PETSC_COMM_WORLD, &scBV);
  CHKERRQ(pcRC);
  pcRC = BVSetOptionsPrefix(scBV, "chkr_");
  CHKERRQ(pcRC);

  if (localSolve) { // OK: we get a sequential dense matrix: this type is supported by BVOrthogonalize.
    pcRC = BVCreateFromMat(pcM, &scBV);
    CHKERRQ(pcRC);
  }
  else { // We get a MPI sparse matrix (not supported by BVOrthogonalize): we must convert it to MPI dense.
    Mat pcMDense;
    pcRC = MatConvert(pcM, MATMPIDENSE, MAT_INITIAL_MATRIX, &pcMDense);
    CHKERRQ(pcRC);
    pcRC = BVCreateFromMat(pcMDense, &scBV);
    CHKERRQ(pcRC);
    pcRC = MatDestroy(&pcMDense);
    CHKERRQ(pcRC);
  }
  pcRC = BVSetType(scBV, BVMAT);
  CHKERRQ(pcRC);
  pcRC = BVSetOrthogonalization(scBV, BV_ORTHOG_MGS, BV_ORTHOG_REFINE_ALWAYS, 0.5, BV_ORTHOG_BLOCK_GS);
  CHKERRQ(pcRC);

  // Orthogonalize the whole set of eigen vectors.

  Mat pcR; // R must be pcNbEV x pcNbEV.
  pcRC = MatCreateSeqDense(PETSC_COMM_SELF, pcNbEV, pcNbEV, NULL, &pcR);
  CHKERRQ(pcRC);

  pcRC = BVSetFromOptions(scBV); // Just before orthogonalize to override default options.
  CHKERRQ(pcRC);
  pcRC = BVOrthogonalize(scBV, pcR); // Computes Z = Q*R, where Q overwrites Z.
  CHKERRQ(pcRC);

  // Check rank: all diagonal values of R must be non zero.

  PetscViewer pcView;
  string checkFile = gCtx->checkFile + ".setup.Z.R";
  pcRC = createViewer(gCtx->checkBin, gCtx->checkMat, PETSC_COMM_SELF, checkFile, pcView);
  CHKERRQ(pcRC);
  pcRC = MatView(pcR, pcView); // pcR is always created with PETSC_COMM_SELF.
  CHKERRQ(pcRC);
  pcRC = PetscViewerDestroy(&pcView);
  CHKERRQ(pcRC);

  double const eps = numeric_limits<double>::epsilon();
  for (PetscInt pcIdx = 0; pcIdx < pcNbEV; pcIdx++) {
    PetscScalar pcVal;
    pcRC = MatGetValues(pcR, 1, &pcIdx, 1, & pcIdx, &pcVal);
    CHKERRQ(pcRC);
    if (fabs(pcVal) <= eps) {
      stringstream msg; msg << "GenEO - check rank: Z = Q*R with R(" << pcIdx << ", " << pcIdx << ") = " << pcVal;
      SETERRABT(msg.str().c_str());
    }
  }

  // Clean.

  pcRC = MatDestroy(&pcR);
  CHKERRQ(pcRC);
  pcRC = BVDestroy(&scBV);
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode fillZE2L(geneoContext const * const gCtx, vector<Vec> const & pcAllEigVecLoc, Mat & pcZE2L) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Fill the local part of the coarse space Z.

  vector<PetscInt> pcIdxColLoc;
  pcIdxColLoc.reserve(gCtx->nbDOFLoc);
  for (unsigned int i = 0; i < gCtx->nbDOFLoc; i++) pcIdxColLoc.push_back(i);

  unsigned int nbEVLoc = pcAllEigVecLoc.size();
  for (PetscInt pcIdx = 0; (unsigned int) pcIdx < nbEVLoc; pcIdx++) {
    Vec pcEigenVecLoc = pcAllEigVecLoc[pcIdx];
    PetscErrorCode pcRC = VecPointwiseMult(pcEigenVecLoc, gCtx->pcDLoc, pcEigenVecLoc); // (7.49) from R1.
    CHKERRQ(pcRC);
    PetscScalar * pcEigenVecArray = NULL;
    pcRC = VecGetArray(pcEigenVecLoc, &pcEigenVecArray);
    CHKERRQ(pcRC);
    pcRC = MatSetValues(pcZE2L, gCtx->nbDOFLoc, pcIdxColLoc.data(), 1, &pcIdx, pcEigenVecArray, INSERT_VALUES);
    CHKERRQ(pcRC);
    pcRC = VecRestoreArray(pcEigenVecLoc, &pcEigenVecArray);
    CHKERRQ(pcRC);
    pcRC = VecDestroy(&pcEigenVecLoc); // Matching VecCreate in eigenLocalSolve / eigenLocalProblem.
    CHKERRQ(pcRC);
  }
  PetscErrorCode pcRC = MatAssemblyBegin(pcZE2L, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);
  pcRC = MatAssemblyEnd(pcZE2L, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);

  // Check on demand.

  if (gCtx->check) {
    pcRC = checkRank(true, pcZE2L, nbEVLoc, gCtx);
    CHKERRQ(pcRC);
  }

  return 0;
}

PetscErrorCode createZE2GOff(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Compute local copy of (distributed) Z.

  PetscInt pcNbDOF = 0, pcNbEV = 0;
  PetscErrorCode pcRC = MatGetSize(gCtx->pcZE2G, &pcNbDOF, &pcNbEV);
  CHKERRQ(pcRC);
  vector<PetscInt> pcIdxEV; pcIdxEV.reserve(pcNbEV);
  for (PetscInt pcIdx = 0; pcIdx < pcNbEV; pcIdx++) pcIdxEV.push_back(pcIdx);
  IS pcISEV;
  pcRC = ISCreateGeneral(PETSC_COMM_SELF, pcNbEV, pcIdxEV.data(), PETSC_COPY_VALUES, &pcISEV);
  CHKERRQ(pcRC);
  vector<PetscInt> pcIdxDOF; pcIdxDOF.reserve(pcNbDOF);
  for (PetscInt pcIdx = 0; pcIdx < pcNbDOF; pcIdx++) pcIdxDOF.push_back(pcIdx);
  IS pcISDOF;
  pcRC = ISCreateGeneral(PETSC_COMM_SELF, pcNbDOF, pcIdxDOF.data(), PETSC_COPY_VALUES, &pcISDOF);
  CHKERRQ(pcRC);
  boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
  PetscInt nbMat = (petscWorld.rank() == 0) ? 1 : 0; // Only the master needs a local copy.
  pcRC = MatCreateSubMatrices(gCtx->pcZE2G, nbMat, &pcISDOF, &pcISEV, MAT_INITIAL_MATRIX, &(gCtx->pcZE2GOff));
  CHKERRQ(pcRC); // Use MatCreateSubMatrices to get sequential matrix (MatCreateSubMatrix would return distributed matrix).

  // Clean.

  pcRC = ISDestroy(&pcISDOF);
  CHKERRQ(pcRC);
  pcRC = ISDestroy(&pcISEV);
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode createZE2L(vector<double> const & pcAllEigValLoc, vector<Vec> const & pcAllEigVecLoc, geneoContext const * const gCtx,
                          Mat & pcZE2L) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Create the local part of the coarse space Z.

  PetscErrorCode pcRC = preallocateZE2L(pcAllEigVecLoc.size(), gCtx, pcZE2L);
  CHKERRQ(pcRC);
  pcRC = fillZE2L(gCtx, pcAllEigVecLoc, pcZE2L);
  CHKERRQ(pcRC);

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = gCtx->debugFile + ".setup.Z";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_SELF, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = MatView(pcZE2L, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);

    debugFile = gCtx->debugFile + ".setup.Z.ev.log";
    ofstream debug(debugFile);
    debug << endl << "Z - nb of eigen values: " << pcAllEigValLoc.size() << endl;
    for (unsigned int ev = 0; ev < pcAllEigValLoc.size(); ev++) {
      debug << "Z - eigen value " << ev << ": " << pcAllEigValLoc[ev] << endl;
    }
  }

  return 0;
}

PetscErrorCode createZE2G(vector<Vec> const & pcAllEigVecLoc, Mat & pcZE2L, geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Get global and local number of eigen values / vectors (for each domain).

  vector<PetscInt> pcLsOfNbEVLoc;
  PetscInt pcNbEVLoc = pcAllEigVecLoc.size();
  boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
  boost::mpi::all_gather(petscWorld, pcNbEVLoc, pcLsOfNbEVLoc);

  PetscInt pcNbEV = 0; // Number of global eigen values / vectors.
  for (auto nbEVLoc = pcLsOfNbEVLoc.cbegin(); nbEVLoc != pcLsOfNbEVLoc.cend(); nbEVLoc++) pcNbEV += *nbEVLoc;

  // Create mapping for eigen values / vectors.

  PetscInt pcIdxEVRootLoc = 0; // Global index where the local eigen vector set begins.
  int rank = petscWorld.rank();
  for (int i = 0; i < rank; i++) pcIdxEVRootLoc += pcLsOfNbEVLoc[i];
  vector<PetscInt> pcIdxEVLoc;
  pcIdxEVLoc.reserve(pcNbEVLoc);
  for (PetscInt pcIdx = 0; pcIdx < pcNbEVLoc; pcIdx++) pcIdxEVLoc.push_back(pcIdxEVRootLoc + pcIdx);

  ISLocalToGlobalMapping evMap;
  PetscErrorCode pcRC = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, pcNbEVLoc, pcIdxEVLoc.data(), PETSC_COPY_VALUES, &evMap);
  CHKERRQ(pcRC);

  // Create the coarse space Z (globally by aggregation of its local parts).

  // MatIS means that the matrix is not assembled. The easiest way to think of this is that processes do not have to hold
  // full rows. One process can hold part of row i, and another processes can hold another part. The local size here is
  // not the size of the local IS block, since that is a property only of MatIS. It is the size of the local piece of the
  // vector you multiply. This allows PETSc to understand the parallel layout of the Vec, and how it matched the Mat.
  Mat pcZE2G;
  PetscInt m;
  pcRC = MatGetLocalSize(gCtx->pcA, &m, NULL);
  CHKERRQ(pcRC);
  pcRC = MatCreateIS(PETSC_COMM_WORLD, 1, m, pcNbEVLoc, gCtx->nbDOF, pcNbEV, gCtx->pcMap, evMap, &pcZE2G);
  CHKERRQ(pcRC);
  pcRC = MatISSetLocalMat(pcZE2G, pcZE2L); // Set domain matrix locally.
  CHKERRQ(pcRC);
  pcRC = MatDestroy(&pcZE2L);
  CHKERRQ(pcRC);
  pcRC = ISLocalToGlobalMappingDestroy(&evMap);
  CHKERRQ(pcRC);

  pcRC = MatAssemblyBegin(pcZE2G, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);
  pcRC = MatAssemblyEnd  (pcZE2G, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);

  // Store aside a MPI version of Z in the context (for later use during iterations).

  pcRC = MatConvert(pcZE2G, MATAIJ, MAT_INITIAL_MATRIX, &(gCtx->pcZE2G)); // Assemble local parts of Z.
  CHKERRQ(pcRC);

  // Check on demand.

  if (gCtx->check) {
    pcRC = checkRank(false, gCtx->pcZE2G, pcNbEV, gCtx);
    CHKERRQ(pcRC);
  }

  // Offload Z on demand.

  if (gCtx->offload) {
    pcRC = createZE2GOff(gCtx);
    CHKERRQ(pcRC);
  }

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = "debug.setup.Z.MatIS";
    pcRC = createViewer(false, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView); // MatIS crashes on binary export...
    CHKERRQ(pcRC);
    pcRC = MatView(pcZE2G, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
    debugFile = "debug.setup.Z.MatMPI";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = MatView(gCtx->pcZE2G, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  // Clean.

  pcRC = MatDestroy(&pcZE2G);
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode getInertia(bool const & localSolve, Mat const & pcM,
                          PetscInt & pcNbNegEV, PetscInt & pcNbNullEV, PetscInt & pcNbPosEV) {
  pcNbNegEV = pcNbNullEV = pcNbPosEV = 0;

  // Apply Sylvester's law: compute inertia.

  PetscErrorCode pcRC = MatSetOption(pcM, MAT_SYMMETRIC, PETSC_TRUE); // Inertia make sense only for symmetric matrices.
  CHKERRQ(pcRC);
  KSP pcKSPLoc;
  if (localSolve) pcRC = KSPCreate(PETSC_COMM_SELF,  &pcKSPLoc); // Local  solve (delegated to each MPI proc).
  else            pcRC = KSPCreate(PETSC_COMM_WORLD, &pcKSPLoc); // Global solve (shared    by  all MPI proc).
  CHKERRQ(pcRC);
  pcRC = KSPSetOptionsPrefix(pcKSPLoc, "syl2_");
  CHKERRQ(pcRC);
  pcRC = KSPSetOperators(pcKSPLoc, pcM, pcM);
  CHKERRQ(pcRC);
  pcRC = KSPSetType(pcKSPLoc, KSPPREONLY); // Direct solve: apply only preconditioner (not A).
  CHKERRQ(pcRC);
  PC pcPCLoc;
  pcRC = KSPGetPC(pcKSPLoc, &pcPCLoc);
  CHKERRQ(pcRC);
  pcRC = PCSetType(pcPCLoc, PCCHOLESKY); // We know the problem is real symetric: it can be decomposed into LDLt.
  CHKERRQ(pcRC);
  pcRC = PCFactorSetMatSolverType(pcPCLoc, "mumps");
  CHKERRQ(pcRC);
  pcRC = KSPSetFromOptions(pcKSPLoc); // Just before setup to override default options.
  CHKERRQ(pcRC);

  pcRC = KSPSetUp(pcKSPLoc);
  CHKERRQ(pcRC);
  PCType pcPCTypeLoc;
  pcRC = PCGetType(pcPCLoc, &pcPCTypeLoc); // May has been changed by the command line.
  CHKERRQ(pcRC);
  if (!pcPCTypeLoc || string(pcPCTypeLoc) != "cholesky") SETERRABT("inertia KO without cholesky");
  Mat pcFactorLoc;
  pcRC = PCFactorGetMatrix(pcPCLoc, &pcFactorLoc);
  CHKERRQ(pcRC);
  pcRC = tuneSolver(pcPCLoc, pcFactorLoc, true);
  CHKERRQ(pcRC);
  pcRC = MatGetInertia(pcFactorLoc, &pcNbNegEV, &pcNbNullEV, &pcNbPosEV);
  CHKERRQ(pcRC);

  // Clean.

  pcRC = KSPDestroy(&pcKSPLoc);
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode estimateNumberOfEigenValues(Mat const & pcALoc, Mat const & pcBLoc,
                                           double const & geneoParam, string const & geneoPb,
                                           geneoContext * const gCtx, PetscInt & pcEstimNbEVLoc) {
  pcEstimNbEVLoc = 0;
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Compute pcALoc - param*pcBLoc to apply Sylvester's law of inertia.

  Mat pcSylvesterLoc; // Matrix used to apply Sylvester's law of inertia.
  PetscErrorCode pcRC = MatDuplicate(pcBLoc, MAT_COPY_VALUES, &pcSylvesterLoc);
  CHKERRQ(pcRC);
  pcRC = MatScale(pcSylvesterLoc, -1.*geneoParam);
  CHKERRQ(pcRC);
  pcRC = MatAXPY(pcSylvesterLoc, 1., pcALoc, DIFFERENT_NONZERO_PATTERN); // pcSylvesterLoc = pcALoc - param*pcBLoc.
  CHKERRQ(pcRC);

  // Compute inertia. The Sylvester matrix may be close to singular (built from the singular Neumann matrix).

  PetscInt pcNbNegEV = 0, pcNbNullEV = 0, pcNbPosEV = 0;
  auto start = chrono::high_resolution_clock::now();
  pcRC = getInertia(true, pcSylvesterLoc, pcNbNegEV, pcNbNullEV, pcNbPosEV);
  CHKERRQ(pcRC);
  auto stop = chrono::high_resolution_clock::now();
  gCtx->lvl2SetupSylTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
  if (geneoPb == "tau")   gCtx->lvl2SetupTauSylTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
  if (geneoPb == "gamma") gCtx->lvl2SetupGammaSylTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  if (geneoPb == "tau")   pcEstimNbEVLoc = pcNbNegEV; // Number of eigen values lower than tau.
  if (geneoPb == "gamma") pcEstimNbEVLoc = pcNbPosEV; // Number of eigen values upper than gamma.
  if ((unsigned int) pcEstimNbEVLoc > gCtx->nbDOFLoc) pcEstimNbEVLoc = gCtx->nbDOFLoc; // Cut-off: there is at most nbDOFLoc eigen values.
  if (gCtx->cut > 0 && pcEstimNbEVLoc > gCtx->cut) pcEstimNbEVLoc = gCtx->cut;
  gCtx->estimDimELoc += pcEstimNbEVLoc;

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = gCtx->debugFile + ".setup." + geneoPb + ".sylvester";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_SELF, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = MatView(pcSylvesterLoc, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);

    debugFile = gCtx->debugFile + ".setup." + geneoPb + ".sylvester.inertia.log";
    ofstream debug(debugFile.c_str());
    debug << endl;
    debug << "nbNegEV " << pcNbNegEV << ", nbNullEV " << pcNbNullEV << ", nbPosEV " << pcNbPosEV << " => estim " << pcEstimNbEVLoc;
    debug << endl;
  }

  // Clean.

  pcRC = MatDestroy(&pcSylvesterLoc);
  CHKERRQ(pcRC);

  return 0;
}

string getConvergedReason(EPSConvergedReason const & pcReason) {
  string reason = "";

  /* converged */
  if (pcReason == EPS_CONVERGED_TOL         ) reason = "EPS_CONVERGED_TOL";
  if (pcReason == EPS_CONVERGED_USER        ) reason = "EPS_CONVERGED_USER";
  if (pcReason == EPS_CONVERGED_ITERATING   ) reason = "EPS_CONVERGED_ITERATING";
  /* diverged */
  if (pcReason == EPS_DIVERGED_ITS          ) reason = "EPS_DIVERGED_ITS";
  if (pcReason == EPS_DIVERGED_BREAKDOWN    ) reason = "EPS_DIVERGED_BREAKDOWN";
  if (pcReason == EPS_DIVERGED_SYMMETRY_LOST) reason = "EPS_DIVERGED_SYMMETRY_LOST";

  return reason;
}

PetscErrorCode checkEPSSolve(EPS const & scEPSLoc, geneoContext * const gCtx, string const & info) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Get reason.

  EPSConvergedReason pcReason;
  PetscErrorCode pcRC = EPSGetConvergedReason(scEPSLoc, &pcReason);
  CHKERRQ(pcRC);
  string reason = getConvergedReason(pcReason);

  // Debug on demand or failure.

  if (pcReason < 0 || gCtx->debug >= 2) {
    if (gCtx->debugFile.length() == 0) { // Failure but no debug.
      boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
      stringstream size; size << petscWorld.size();
      stringstream rank; rank << setfill('0') << setw(size.str().length()) << petscWorld.rank();
      gCtx->debugFile = "debug" + rank.str();
    }

    PetscViewer pcView;
    string debugEPSFile = gCtx->debugFile + ".setup." + info + ".eps";
    pcRC = createViewer(false, false, PETSC_COMM_SELF, debugEPSFile, pcView);
    CHKERRQ(pcRC);
    pcRC = EPSView(scEPSLoc, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);

    BV scBVLoc;
    pcRC = EPSGetBV(scEPSLoc, &scBVLoc);
    CHKERRQ(pcRC);
    string debugBVFile = gCtx->debugFile + ".setup." + info + ".bv";
    pcRC = createViewer(false, false, PETSC_COMM_SELF, debugBVFile, pcView);
    CHKERRQ(pcRC);
    pcRC = BVView(scBVLoc, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  if (pcReason < 0) {
    stringstream msg; msg << "GenEO preconditioner: " << info << " KO (" << reason << ")";
    SETERRABT(msg.str().c_str());
  }

  return 0;
}

PetscErrorCode eigenLocalSolve(geneoContext * const gCtx, EPS & scEPSLoc, double const & geneoParam, string const & geneoPb,
                               vector<PetscScalar> & pcEigValLoc, vector<Vec> & pcEigVecLoc) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Set target: by default the shift is equal to the target. The target is where you want to find eigenvalues.
  // Beforehand, there is no way to know exactly what the eigen values will be. If it turns out that the target
  // is exactly equal to an eigenvalue, then A-sigma*I or A-sigma*B could be singular: the solve may break.
  // In this case, afterwards, you can change slightly the shift (sigma) to avoid the solve break down.

  if (geneoPb == "tau") {
    PetscErrorCode pcRC = EPSSetInterval(scEPSLoc, 0., geneoParam); // Look for eigen values in [0., tau].
    CHKERRQ(pcRC);
    pcRC = EPSSetTarget(scEPSLoc, 0.); // Target = shift = 0. to get the smallest eigen values.
    CHKERRQ(pcRC);
    pcRC = EPSSetWhichEigenpairs(scEPSLoc, EPS_TARGET_MAGNITUDE);
    CHKERRQ(pcRC);
    // Idea: finding the bigger (resp. smaller) eigen values is "easy" (resp. "difficult").
    //       without shift-invert: look for smallest values of the  initial problem => "difficult and slow".
    //       with    shift-invert: look for biggest  values of the inverted problem => "easy and fast".
    //       with or without shift-invert, the same eigen values are computed (small <=> "inverted" big) but efficiency differs !
    ST scSTLoc;
    pcRC = EPSGetST(scEPSLoc, &scSTLoc);
    CHKERRQ(pcRC);
    pcRC = STSetType(scSTLoc, STSINVERT); // Caution: MUST be used with EPS_TARGET_MAGNITUDE.
    CHKERRQ(pcRC);
  }
  else if (geneoPb == "gamma") {
    PetscErrorCode pcRC = EPSSetInterval(scEPSLoc, geneoParam, PETSC_MAX_REAL); // Look for eigen values in [gamma, +infinity].
    CHKERRQ(pcRC);
    pcRC = EPSSetWhichEigenpairs(scEPSLoc, EPS_LARGEST_MAGNITUDE); // Get largest eigen values.
    CHKERRQ(pcRC);
  }
  PetscErrorCode pcRC = EPSSetTolerances(scEPSLoc, 1.e-03, PETSC_DEFAULT); // Only need "good enough" eigen vectors approximations.
  CHKERRQ(pcRC);

  // Solve the generalized eigen value problem.

  pcRC = EPSSetType(scEPSLoc, "arpack");
  CHKERRQ(pcRC);
  pcRC = EPSSetProblemType(scEPSLoc, EPS_GHEP); // We know the problem is real symmetric.
  CHKERRQ(pcRC);
  pcRC = EPSSetFromOptions(scEPSLoc); // Just before setup to override default options.
  CHKERRQ(pcRC);

  auto start = chrono::high_resolution_clock::now();
  pcRC = EPSSetUp(scEPSLoc);
  CHKERRQ(pcRC);

  ST scSTLoc;
  pcRC = EPSGetST(scEPSLoc, &scSTLoc);
  CHKERRQ(pcRC);
  KSP pcSTKSPLoc;
  pcRC = STGetKSP(scSTLoc, &pcSTKSPLoc);
  CHKERRQ(pcRC);
  PC pcSTKSPPCLoc;
  pcRC = KSPGetPC(pcSTKSPLoc, &pcSTKSPPCLoc);
  CHKERRQ(pcRC);
  PCType pcSTKSPPCTypeLoc;
  pcRC = PCGetType(pcSTKSPPCLoc, &pcSTKSPPCTypeLoc); // May has been changed by the command line.
  CHKERRQ(pcRC);
  if (pcSTKSPPCTypeLoc && string(pcSTKSPPCTypeLoc) == "lu") {
    Mat pcSTFactorLoc;
    pcRC = PCFactorGetMatrix(pcSTKSPPCLoc, &pcSTFactorLoc); // Force LU factorisation (stand by for later use).
    CHKERRQ(pcRC);
    pcRC = tuneSolver(pcSTKSPPCLoc, pcSTFactorLoc); // Tune solver after setup, but, before solve.
    CHKERRQ(pcRC);
  }

  start = chrono::high_resolution_clock::now();
  pcRC = EPSSolve(scEPSLoc);
  CHKERRQ(pcRC);
  pcRC = checkEPSSolve(scEPSLoc, gCtx, "els2-" + geneoPb);
  CHKERRQ(pcRC);
  auto stop = chrono::high_resolution_clock::now();
  gCtx->lvl2SetupEigTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
  if (geneoPb == "tau")   gCtx->lvl2SetupTauEigTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
  if (geneoPb == "gamma") gCtx->lvl2SetupGammaEigTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  // Retrieve solutions of the eigen value problem.

  PetscInt pcNbEVConvLoc = 0;
  pcRC = EPSGetConverged(scEPSLoc, &pcNbEVConvLoc);
  CHKERRQ(pcRC);
  for (PetscInt pcIdx = 0; pcIdx < pcNbEVConvLoc; pcIdx++) {
    PetscScalar pcEigenValLoc = 0.;
    pcRC = EPSGetEigenvalue(scEPSLoc, pcIdx, &pcEigenValLoc, NULL);
    CHKERRQ(pcRC);
    if (geneoPb == "tau"   && pcEigenValLoc > geneoParam) continue; // Get eigen values in [0, tau].
    if (geneoPb == "gamma" && pcEigenValLoc < geneoParam) continue; // Get eigen values in [gamma, +infinity].
    Vec pcEigenVecLoc;
    PetscErrorCode pcRC = VecCreateSeq(PETSC_COMM_SELF, gCtx->nbDOFLoc, &pcEigenVecLoc);
    CHKERRQ(pcRC); // Matching VecDestroy in fillZE2L.
    pcRC = EPSGetEigenvector(scEPSLoc, pcIdx, pcEigenVecLoc, NULL);
    CHKERRQ(pcRC);
    pcEigValLoc.push_back(pcEigenValLoc);
    pcEigVecLoc.push_back(pcEigenVecLoc);
  }

  // Debug on demand.

  if (gCtx->debug >= 2) {
    ofstream debug(gCtx->debugFile + ".setup." + geneoPb + ".ev.log");
    if (geneoPb == "tau")   debug << "tau "   << gCtx->tau   << ", tauLoc "   << gCtx->tauLoc   << endl;
    if (geneoPb == "gamma") debug << "gamma " << gCtx->gamma << ", gammaLoc " << gCtx->gammaLoc << endl;
    debug << endl << geneoPb << " - nb of eigen values added to Z: " << pcEigValLoc.size() << endl;
    for (unsigned int ev = 0; ev < pcEigValLoc.size(); ev++) {
      debug << geneoPb << " - eigen value " << ev << " added to Z: " << pcEigValLoc[ev] << endl;
    }
    debug << endl << geneoPb << " - nb of eigen values found (candidate for Z): " << pcNbEVConvLoc << endl;
    for (PetscInt pcIdx = 0; pcIdx < pcNbEVConvLoc; pcIdx++) {
      PetscScalar pcEigenValLoc = 0.;
      pcRC = EPSGetEigenvalue(scEPSLoc, pcIdx, &pcEigenValLoc, NULL);
      CHKERRQ(pcRC);
      debug << geneoPb << " - eigen value " << pcIdx << " found (candidate for Z): " << pcEigenValLoc << endl;
    }
  }

  return 0;
}

PetscErrorCode buildEigenSolver(EPS & scEPSLoc, char const * prefix, bool const localSolve = true) {
  // Create eigen solver.

  PetscErrorCode pcRC;
  if (localSolve) pcRC = EPSCreate(PETSC_COMM_SELF,  &scEPSLoc); // Local  solve (delegated to each MPI proc).
  else            pcRC = EPSCreate(PETSC_COMM_WORLD, &scEPSLoc); // Global solve (shared    by  all MPI proc).
  CHKERRQ(pcRC);
  if (prefix) {
    pcRC = EPSSetOptionsPrefix(scEPSLoc, prefix);
    CHKERRQ(pcRC);
  }

  // Get KSP of the ST.

  ST scSTLoc;
  pcRC = EPSGetST(scEPSLoc, &scSTLoc);
  CHKERRQ(pcRC);
  KSP pcSTKSPLoc;
  pcRC = STGetKSP(scSTLoc, &pcSTKSPLoc);
  CHKERRQ(pcRC);
  pcRC = KSPSetType(pcSTKSPLoc, KSPPREONLY); // Direct solve: apply only preconditioner (not A).
  CHKERRQ(pcRC);

  // Get the PC of the ST: set default to mumps.

  PC pcSTKSPPCLoc;
  pcRC = KSPGetPC(pcSTKSPLoc, &pcSTKSPPCLoc);
  CHKERRQ(pcRC);
  pcRC = PCSetType(pcSTKSPPCLoc, PCLU); // Direct solve: preconditioner "of type LU" <=> direct solver.
  CHKERRQ(pcRC);
  pcRC = PCFactorSetMatSolverType(pcSTKSPPCLoc, "mumps");
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode checkSPD(bool const & localSolve, Mat const & pcM, string const & baseFileName, string const & info) {
  // Compute smallest eigen values.

  EPS scEPSLoc; // Solve local coarse space (eigen) problem.
  PetscErrorCode pcRC = buildEigenSolver(scEPSLoc, "chks_", localSolve);
  CHKERRQ(pcRC);
  pcRC = EPSSetOperators(scEPSLoc, pcM, NULL); // M*V = lambda*V
  CHKERRQ(pcRC);
  pcRC = EPSSetWhichEigenpairs(scEPSLoc, EPS_SMALLEST_MAGNITUDE); // Get the smallest eigen values.
  CHKERRQ(pcRC);
  pcRC = EPSSetType(scEPSLoc, "arpack");
  CHKERRQ(pcRC);
  pcRC = EPSSetFromOptions(scEPSLoc); // Just before setup to override default options.
  CHKERRQ(pcRC);
  pcRC = EPSSolve(scEPSLoc);
  CHKERRQ(pcRC);

  // As the matrix must be real SPD: check symmetric => diagonisable, check positive => all eigen values > 0.

  PetscInt pcNbEVConvLoc = 0;
  pcRC = EPSGetConverged(scEPSLoc, &pcNbEVConvLoc);
  CHKERRQ(pcRC);
  string checkFile = baseFileName + ".SPD." + info + ".log";
  ofstream check(checkFile.c_str());
  if (pcNbEVConvLoc < 1) {
    check << "CHECK: " << info << " SPD - solve didn't find any solution" << endl;
  }
  else {
    double const eps = numeric_limits<double>::epsilon();
    for (PetscInt pcIdx = 0; pcIdx < pcNbEVConvLoc; pcIdx++) {
      PetscScalar pcEigenVal = 0.;
      pcRC = EPSGetEigenvalue(scEPSLoc, pcIdx, &pcEigenVal, NULL);
      CHKERRQ(pcRC);
      check << info << " - eigen value " << pcIdx << ": " << pcEigenVal << endl;
      if (fabs(pcEigenVal) <= eps) { // The smallest eigen value must be positive.
        stringstream msg; msg << "GenEO - check SPD: " << info << " not SPD, bad eigen value " << pcEigenVal;
        SETERRABT(msg.str().c_str());
      }
    }
  }

  // Compute inertia.

  PetscInt pcNbNegEV = 0, pcNbNullEV = 0, pcNbPosEV = 0;
  pcRC = getInertia(localSolve, pcM, pcNbNegEV, pcNbNullEV, pcNbPosEV);
  CHKERRQ(pcRC);
  check << endl << info << " - inertia: nbNegEV " << pcNbNegEV << ", nbNullEV " << pcNbNullEV << ", nbPosEV " << pcNbPosEV << endl;
  if (pcNbNegEV > 0 || pcNbNullEV > 0) {
    stringstream msg; msg << "GenEO - check SPD: not SPD (inertia - negative or null eigen value found)";
    SETERRABT(msg.str().c_str());
  }

  // Clean.

  pcRC = EPSDestroy(&scEPSLoc);
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode eigenLocalProblem(EPS & scEPSLoc, Mat const & pcALoc, Mat const & pcBLoc,
                                 double const & geneoParam, string const & geneoPb, geneoContext * const gCtx,
                                 vector<double> & pcAllEigValLoc, vector<Vec> & pcAllEigVecLoc) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Estimate number of eigen values on demand.

  vector<PetscScalar> pcEigValLoc; // List of local eigen values.
  vector<Vec> pcEigVecLoc; // List of local eigen vectors.
  PetscErrorCode pcRC;
  if (!gCtx->noSyl) { // Use Sylvester's law.
    // Compute only the (smallest/highest) eigen values relatively to param (to build a coarse space Z).

    PetscInt pcEstimNbEVLoc = 0;
    pcRC = estimateNumberOfEigenValues(pcALoc, pcBLoc, geneoParam, geneoPb, gCtx, pcEstimNbEVLoc);
    CHKERRQ(pcRC);

    // Set dimensions.

    if (pcEstimNbEVLoc > 0) {
      pcRC = EPSSetDimensions(scEPSLoc, pcEstimNbEVLoc, PETSC_DEFAULT, PETSC_DEFAULT);
      CHKERRQ(pcRC);
      pcEigValLoc.reserve(pcEstimNbEVLoc);
      pcEigVecLoc.reserve(pcEstimNbEVLoc);
    }
  }

  // Limit the number of eigen values on demand.

  if (gCtx->cut > 0) {
    PetscInt pcNbEVLoc = 0;
    pcRC = EPSGetDimensions(scEPSLoc, &pcNbEVLoc, NULL, NULL);
    CHKERRQ(pcRC);
    if (pcNbEVLoc > gCtx->cut) {
      pcRC = EPSSetDimensions(scEPSLoc, gCtx->cut, PETSC_DEFAULT, PETSC_DEFAULT);
      CHKERRQ(pcRC);
    }
  }

  // Check on demand.

  if (gCtx->check) {
    pcRC = checkSPD(true, pcBLoc, gCtx->checkFile, geneoPb + ".B");
    CHKERRQ(pcRC);
  }

  // Solve the eigen problem.

  pcRC = EPSSetOperators(scEPSLoc, pcALoc, pcBLoc); // A*V = lambda*B*V
  CHKERRQ(pcRC);
  pcRC = eigenLocalSolve(gCtx, scEPSLoc, geneoParam, geneoPb, pcEigValLoc, pcEigVecLoc);
  CHKERRQ(pcRC);

  // Use constant vector if 0. has not been found to be an eigen value.

  if (geneoPb == "tau") { // Look for eigen values lesser than tau.
    double const eps = numeric_limits<double>::epsilon();
    bool noZeroEigVal = (pcEigValLoc.size() > 0 && *min_element(begin(pcEigValLoc), end(pcEigValLoc)) >= eps) ? true : false;
    if (noZeroEigVal) {
      Vec pcNicolaidesVecLoc;
      pcRC = VecCreateSeq(PETSC_COMM_SELF, gCtx->nbDOFLoc, &pcNicolaidesVecLoc);
      CHKERRQ(pcRC);
      pcRC = VecSet(pcNicolaidesVecLoc, 1.); // Rigid body motion (associated to eigen value 0.).
      CHKERRQ(pcRC);
      Vec pcVecLoc;
      pcRC = VecCreateSeq(PETSC_COMM_SELF, gCtx->nbDOFLoc, &pcVecLoc);
      CHKERRQ(pcRC);
      pcRC = MatMult(pcALoc, pcNicolaidesVecLoc, pcVecLoc);
      CHKERRQ(pcRC);
      PetscReal pcNumerator = 0.;
      pcRC = VecDot(pcVecLoc, pcNicolaidesVecLoc, &pcNumerator);
      CHKERRQ(pcRC);
      pcRC = MatMult(pcBLoc, pcNicolaidesVecLoc, pcVecLoc);
      CHKERRQ(pcRC);
      PetscReal pcDenominator = 0.;
      pcRC = VecDot(pcVecLoc, pcNicolaidesVecLoc, &pcDenominator);
      CHKERRQ(pcRC);
      pcRC = VecDestroy(&pcVecLoc);
      CHKERRQ(pcRC);

      // Debug on demand.

      double ratio = fabs(pcNumerator/pcDenominator);
      bool addNicolaides = (ratio <= numeric_limits<float>::epsilon()) ? true : false;
      if (gCtx->debug >= 2) {
        ofstream debug(gCtx->debugFile + ".setup." + geneoPb + ".nicolaides.log");
        debug << geneoPb << " - nicolaides, (AV, V) " << pcNumerator << endl;
        debug << geneoPb << " - nicolaides, (BV, V) " << pcDenominator << endl;
        debug << geneoPb << " - nicolaides, ratio " << ratio << ", added " << addNicolaides << endl;
      }

      // Add constant vector if it is in the kernel of A.

      if (addNicolaides) {
        pcEigValLoc.push_back(0.);
        pcEigVecLoc.push_back(pcNicolaidesVecLoc); // Matching VecDestroy in fillZE2L.
        gCtx->nicolaidesLoc += 1;
      }
      else {
        pcRC = VecDestroy(&pcNicolaidesVecLoc);
        CHKERRQ(pcRC);
      }
    }
    else {
      if (gCtx->debug >= 2) {
        ofstream debug(gCtx->debugFile + ".setup." + geneoPb + ".nicolaides.log");
        debug << geneoPb << " - nicolaides not added: number of eigen value(s) " << pcEigValLoc.size() << endl;
        if (pcEigValLoc.size() > 0) {
          double minEV = *min_element(begin(pcEigValLoc), end(pcEigValLoc));
          debug << geneoPb << " - nicolaides not added: min eigen value " << minEV << endl;
        }
      }
    }
  }

  // Return eigen values and vectors.

  pcAllEigValLoc.insert(pcAllEigValLoc.end(), pcEigValLoc.begin(), pcEigValLoc.end());
  pcAllEigVecLoc.insert(pcAllEigVecLoc.end(), pcEigVecLoc.begin(), pcEigVecLoc.end());

  return 0;
}

PetscErrorCode createPartitionOfUnity(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");
  if (!gCtx->dofIdxMultLoc) SETERRABT("GenEO preconditioner without DOF multiplicity");
  if (gCtx->dofIdxMultLoc->size() != gCtx->nbDOFLoc) SETERRABT("GenEO preconditioner bad DOF multiplicity");

  // Compute partition of unity.

  PetscErrorCode pcRC;
  PetscScalar   *array;
  pcRC = VecCreateSeq(PETSC_COMM_SELF, gCtx->nbDOFLoc, &gCtx->pcDLoc); CHKERRQ(pcRC);
  pcRC = VecGetArray(gCtx->pcDLoc, &array); CHKERRQ(pcRC);

  int i = 0;
  for (auto mult = gCtx->dofIdxMultLoc->cbegin(); mult != gCtx->dofIdxMultLoc->cend(); mult++, i++) {
    array[i] = 1.0/((PetscScalar)*mult);
  }
  pcRC = VecRestoreArray(gCtx->pcDLoc, &array);
  CHKERRQ(pcRC);

  // Debug or check on demand.

  pcRC = VecViewFromOptions(gCtx->pcDLoc, NULL, "-geneo_partition_of_unity_view");
  CHKERRQ(pcRC);
  if (gCtx->check) {
    double const eps = numeric_limits<double>::epsilon();
    PetscReal min;
    pcRC = VecMin(gCtx->pcDLoc, NULL, &min);
    CHKERRQ(pcRC);
    if (fabs(min) <= eps) {
      stringstream msg; msg << "GenEO - check D: bad partition of unity, min " << min;
      SETERRABT(msg.str().c_str());
    }
  }

  return 0;
}

PetscErrorCode createEEigOff(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Compute local copy of (distributed) E.

  PetscInt pcNbEV = 0;
  PetscErrorCode pcRC = MatGetSize(gCtx->pcEEig, &pcNbEV, NULL);
  CHKERRQ(pcRC);
  vector<PetscInt> pcIdxEV; pcIdxEV.reserve(pcNbEV);
  for (PetscInt pcIdx = 0; pcIdx < pcNbEV; pcIdx++) pcIdxEV.push_back(pcIdx);
  IS pcISEV;
  pcRC = ISCreateGeneral(PETSC_COMM_SELF, pcNbEV, pcIdxEV.data(), PETSC_COPY_VALUES, &pcISEV);
  CHKERRQ(pcRC);
  boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
  PetscInt nbMat = (petscWorld.rank() == 0) ? 1 : 0; // Only the master needs a local copy.
  pcRC = MatCreateSubMatrices(gCtx->pcEEig, nbMat, &pcISEV, &pcISEV, MAT_INITIAL_MATRIX, &(gCtx->pcEEigOff));
  CHKERRQ(pcRC); // Use MatCreateSubMatrices to get sequential matrix (MatCreateSubMatrix would return distributed matrix).

  // Clean.

  pcRC = ISDestroy(&pcISEV);
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode createEEig(geneoContext * const gCtx, Mat const & pcA) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Compute E = Z.A.Zt (with A and Z as MPI matrix - not as MatIS).

  PetscErrorCode pcRC = MatPtAP(pcA, gCtx->pcZE2G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(gCtx->pcEEig));
  CHKERRQ(pcRC);

  // Create solver for local coarse-projected problem (LU factorisation).

  PC pcPCL2 = NULL;
  if (gCtx->offload) {
    pcRC = createEEigOff(gCtx);
    CHKERRQ(pcRC);
    boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
    if (petscWorld.rank() == 0) {
      pcRC = KSPCreate(PETSC_COMM_SELF, &(gCtx->pcKSPL2Off)); // Local solve (delegated to each MPI proc).
      CHKERRQ(pcRC);
      pcRC = KSPSetOptionsPrefix(gCtx->pcKSPL2Off, "dcs2_");
      CHKERRQ(pcRC);
      if (!gCtx->pcEEigOff) SETERRABT("GenEO preconditioner: offload E KO");
      pcRC = KSPSetOperators(gCtx->pcKSPL2Off, *(gCtx->pcEEigOff), *(gCtx->pcEEigOff));
      CHKERRQ(pcRC);
      pcRC = directLocalSolve(gCtx, gCtx->pcKSPL2Off); // E^-1.
      CHKERRQ(pcRC);

      pcRC = KSPGetPC(gCtx->pcKSPL2Off, &pcPCL2); // To save informations later on.
      CHKERRQ(pcRC);
    }
  }
  else {
    pcRC = KSPCreate(PETSC_COMM_WORLD, &(gCtx->pcKSPL2)); // Global solve (shared by all MPI proc).
    CHKERRQ(pcRC);
    pcRC = KSPSetOptionsPrefix(gCtx->pcKSPL2, "dcs2_");
    CHKERRQ(pcRC);
    pcRC = KSPSetOperators(gCtx->pcKSPL2, gCtx->pcEEig, gCtx->pcEEig);
    CHKERRQ(pcRC);
    pcRC = directLocalSolve(gCtx, gCtx->pcKSPL2); // E^-1.
    CHKERRQ(pcRC);

    pcRC = KSPGetPC(gCtx->pcKSPL2, &pcPCL2); // To save informations later on.
    CHKERRQ(pcRC);
  }

  // Save information message about level 2 (to be printed in output).

  if (pcPCL2) {
    MatSolverType pcType = NULL;
    pcRC = PCFactorGetMatSolverType(pcPCL2, &pcType);
    CHKERRQ(pcRC);
    if (pcType) gCtx->infoL2 += " " + string(pcType);
  }

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = "debug.setup.E";
    PetscErrorCode pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = MatView(gCtx->pcEEig, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  return 0;
}

double getLocalGenEOTau(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");
  if (!gCtx->dofIdxMultLoc) SETERRABT("GenEO preconditioner without DOF multiplicities");

  // Use constant parameter on demand.

  double tauLoc = gCtx->tau;
  if (gCtx->cst) return tauLoc;

  // Compute maximal multiplicity in the domain.

  unsigned int k = *max_element(gCtx->dofIdxMultLoc->cbegin(), gCtx->dofIdxMultLoc->cend());

  // Compute local tau.

  tauLoc = k*gCtx->tau;
  if (tauLoc >= 1.) tauLoc = 0.9; // Resort to tau "far" from 1.

  gCtx->tauLoc = tauLoc;

  return tauLoc;
}

double getLocalGenEOGamma(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");
  if (!gCtx->intersectLoc) SETERRABT("GenEO preconditioner without local intersection");

  // Use constant parameter on demand.

  double gammaLoc = gCtx->gamma;
  if (gCtx->cst) return gammaLoc;

  // Build the matrix of connections: distributed matrix whose assembly will gather local connection informations.

  boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
  int nbPart = petscWorld.size(); // The number of partitions is the number of processus.
  Mat pcConnect; // Connections = Adjacency + Identity.
  PetscErrorCode pcRC = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nbPart, nbPart, NULL, &pcConnect);
  CHKERRQ(pcRC);
  vector<PetscScalar> pcCij; vector<PetscInt> pcIdx;
  pcCij.reserve(nbPart); pcIdx.reserve(nbPart);
  PetscInt p = petscWorld.rank();
  for (unsigned int q = 0; q < gCtx->intersectLoc->size(); q++) { // Loop over all partitions.
    if (p == (PetscInt) q) pcCij.push_back(1.); // Identity.
    else {                                       // Adjacency.
      vector<unsigned int> const & intersectPQLoc = (*(gCtx->intersectLoc))[q];
      bool intersectPQ = (intersectPQLoc.size() == 0) ? false : true;
      if (intersectPQ) pcCij.push_back(0.); // Partitions p and q do not intersect.
      else             pcCij.push_back(1.); // Partitions p and q do     intersect.
    }
    pcIdx.push_back(q);
  }
  pcRC = MatSetValues(pcConnect, 1, &p, nbPart, pcIdx.data(), pcCij.data(), INSERT_VALUES); // Write the p-th row.
  CHKERRQ(pcRC);
  pcRC = MatAssemblyBegin(pcConnect, MAT_FINAL_ASSEMBLY); // Gather all connection informations (including from the other domains).
  CHKERRQ(pcRC);
  pcRC = MatAssemblyEnd  (pcConnect, MAT_FINAL_ASSEMBLY); // Gather all connection informations (including from the other domains).
  CHKERRQ(pcRC);

  // Build the M matrix: local (sequential) matrix (gathering all global connection informations) created from pcConnect.

  IS pcIS;
  pcRC = ISCreateGeneral(PETSC_COMM_SELF, nbPart, pcIdx.data(), PETSC_COPY_VALUES, &pcIS);
  CHKERRQ(pcRC);
  Mat * pcMLoc = NULL; // M matrix.
  pcRC = MatCreateSubMatrices(pcConnect, 1, &pcIS, &pcIS, MAT_INITIAL_MATRIX, &pcMLoc);
  CHKERRQ(pcRC); // Use MatCreateSubMatrices to get sequential matrix (MatCreateSubMatrix would return distributed matrix).
  pcRC = ISDestroy(&pcIS);
  CHKERRQ(pcRC);
  pcRC = MatDestroy(&pcConnect);
  CHKERRQ(pcRC);

  if (!pcMLoc) SETERRABT("GenEO preconditioner: bad MLoc");
  Vec pcFLoc;
  pcRC = MatCreateVecs(*pcMLoc, &pcFLoc, NULL);
  CHKERRQ(pcRC);
  pcRC = MatGetRowSum(*pcMLoc, pcFLoc); // Fi = sum(Cij).
  CHKERRQ(pcRC);
  pcRC = VecPow(pcFLoc, -1.); // Fi = 1./sum(Cij).
  CHKERRQ(pcRC);
  pcRC = MatDiagonalScale(*pcMLoc, pcFLoc, pcFLoc); // Mij = Cij*Fi*Fj.
  CHKERRQ(pcRC);

  // Compute the largest eignevalue of M.

  EPS scEPSLoc; // Solve local coarse space (eigen) problem.
  pcRC = buildEigenSolver(scEPSLoc, "ubl2_");
  CHKERRQ(pcRC);
  pcRC = EPSSetOperators(scEPSLoc, *pcMLoc, NULL); // M*V = lambda*V
  CHKERRQ(pcRC);
  pcRC = EPSSetDimensions(scEPSLoc, 1, PETSC_DEFAULT, PETSC_DEFAULT); // Get the largest eigen values.
  CHKERRQ(pcRC);
  pcRC = EPSSetWhichEigenpairs(scEPSLoc, EPS_LARGEST_MAGNITUDE); // Get the largest eigen values.
  CHKERRQ(pcRC);
  pcRC = EPSSetTolerances(scEPSLoc, 1.e-03, PETSC_DEFAULT); // Only need "good enough" eigen vectors approximations.
  CHKERRQ(pcRC);
  pcRC = EPSSetType(scEPSLoc, "lapack");
  CHKERRQ(pcRC);
  pcRC = EPSSetProblemType(scEPSLoc, EPS_HEP); // We know the problem is real symmetric.
  CHKERRQ(pcRC);
  pcRC = EPSSetFromOptions(scEPSLoc); // Just before setup to override default options.
  CHKERRQ(pcRC);
  pcRC = EPSSolve(scEPSLoc);
  CHKERRQ(pcRC);
  pcRC = checkEPSSolve(scEPSLoc, gCtx, "ubl2");
  CHKERRQ(pcRC);

  // Compute local gamma.

  PetscInt pcNbEVConvLoc = 0;
  pcRC = EPSGetConverged(scEPSLoc, &pcNbEVConvLoc);
  CHKERRQ(pcRC);
  if (pcNbEVConvLoc < 1) return gammaLoc;
  PetscScalar pcEigenValLoc = 0.;
  pcRC = EPSGetEigenvalue(scEPSLoc, 0, &pcEigenValLoc, NULL);
  CHKERRQ(pcRC);
  gammaLoc = gammaLoc / pcEigenValLoc;
  PetscScalar pcFiLoc;
  pcRC = VecGetValues(pcFLoc, 1, &p, &pcFiLoc);
  CHKERRQ(pcRC);
  gammaLoc = gammaLoc * pcFiLoc * pcFiLoc;
  if (gammaLoc <= 1.) gammaLoc = 1.1; // Resort to gamma "far" from 1.

  gCtx->gammaLoc = gammaLoc;

  // Clean.

  pcRC = EPSDestroy(&scEPSLoc);
  CHKERRQ(pcRC);
  pcRC = VecDestroy(&pcFLoc);
  CHKERRQ(pcRC);
  pcRC = MatDestroyMatrices(1, &pcMLoc);
  CHKERRQ(pcRC);

  return gammaLoc;
}

int buildCoarseSpaceWithGenEO(Mat const & pcANeuLoc, Mat const * const pcADirLoc, Mat const & pcARobLoc,
                              geneoContext * const gCtx, Mat const & pcA) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");
  if (!pcADirLoc) SETERRABT("GenEO preconditioner without dirichlet matrix");

  if (!gCtx->lvl2) return 0; // Nothing to do.

  // Get the Dirichlet matrix weighted by the partition of unity (local matrix).

  Mat pcDADirDLoc;
  PetscErrorCode pcRC = MatDuplicate(*pcADirLoc, MAT_COPY_VALUES, &pcDADirDLoc);
  CHKERRQ(pcRC);
  pcRC = MatDiagonalScale(pcDADirDLoc, gCtx->pcDLoc, gCtx->pcDLoc);
  CHKERRQ(pcRC);

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = gCtx->debugFile + ".setup.DADirD";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_SELF, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = MatView(pcDADirDLoc, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  // Get eigen vectors (to build the coarse space).

  EPS scEPSLoc; // Solve local coarse space (eigen) problem.
  pcRC = buildEigenSolver(scEPSLoc, "els2_");
  CHKERRQ(pcRC);

  vector<double> pcAllEigValLoc; // List of local eigen values.
  vector<Vec> pcAllEigVecLoc; // List of local eigen vectors.
  if (gCtx->lvl2 == 1) { // Apply GenEO-1.
    pcRC = eigenLocalProblem(scEPSLoc, pcANeuLoc, pcDADirDLoc, gCtx->tau, "tau", gCtx, pcAllEigValLoc, pcAllEigVecLoc); // Def 7.14 from R1.
    CHKERRQ(pcRC);
  }
  else if (gCtx->lvl2 == 2) { // Apply GenEO-2.
    if (gCtx->cut >= 2) gCtx->cut = gCtx->cut / 2; // GenEO-2 has 2 eigen problems.

    // Solve the tau eigen problem.

    auto start = chrono::high_resolution_clock::now();
    double tauLoc = getLocalGenEOTau(gCtx);
    auto stop = chrono::high_resolution_clock::now();
    gCtx->lvl2SetupTauLocTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
    pcRC = eigenLocalProblem(scEPSLoc, pcANeuLoc, pcARobLoc, tauLoc, "tau", gCtx, pcAllEigValLoc, pcAllEigVecLoc); // (19) from R4.
    CHKERRQ(pcRC);

    // Destroy, and, recreate an eigen solver: restart from scratch.

    pcRC = EPSDestroy(&scEPSLoc);
    CHKERRQ(pcRC);
    pcRC = buildEigenSolver(scEPSLoc, "els2_");
    CHKERRQ(pcRC);

    // Solve the gamma eigen problem.

    start = chrono::high_resolution_clock::now();
    double gammaLoc = getLocalGenEOGamma(gCtx);
    stop = chrono::high_resolution_clock::now();
    gCtx->lvl2SetupGammaLocTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
    pcRC = eigenLocalProblem(scEPSLoc, pcDADirDLoc, pcARobLoc, gammaLoc, "gamma", gCtx, pcAllEigValLoc, pcAllEigVecLoc); // (21) from R4.
    CHKERRQ(pcRC);
  }

  // Empty Z matrix is not allowed ! Add constant vector (Nicolaides) in this case.

  if (pcAllEigVecLoc.size() == 0) {
    Vec pcNicolaidesVecLoc;
    pcRC = VecCreateSeq(PETSC_COMM_SELF, gCtx->nbDOFLoc, &pcNicolaidesVecLoc);
    CHKERRQ(pcRC);
    pcRC = VecSet(pcNicolaidesVecLoc, 1.); // Rigid body motion (associated to eigen value 0.).
    CHKERRQ(pcRC);
    pcAllEigValLoc.push_back(0.);
    pcAllEigVecLoc.push_back(pcNicolaidesVecLoc); // Matching VecDestroy in fillZE2L.
    gCtx->nicolaidesLoc += 1;
  }

  // Save information message about level 2 (to be printed in output).

  gCtx->realDimELoc = pcAllEigVecLoc.size();
  EPSType pcType = NULL;
  pcRC = EPSGetType(scEPSLoc, &pcType);
  CHKERRQ(pcRC);
  string type = (pcType) ? string((char *) pcType) : "arpack";
  gCtx->infoL2 += type;

  // Synchronisation before creating Z (= global distributed MatIS that MUST be built by all processus).
  // In case all  eigen solves are     cheap, the barrier does not hurt.
  // In case some eigen solves are not cheap, the barrier avoids irrelevant timings:
  //   - Processus where solves are cheap are stuck in the set up of Z (waiting for other processus).
  //   - This waiting time should not be attributed to Z, but instead, should be attributed to eigen solves.

  boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
  petscWorld.barrier();

  // Build coarse space with eigen vectors.
  // Idea: the (large) wave lengths that slow down convergence are "put" into a coarse space.
  //   1. The coarse space (E) is solved with  a    direct method (smallest part of all wave lengths).
  //   2. The rest             is solved with an iterative method (biggest  part of all wave lengths).

  auto start = chrono::high_resolution_clock::now();
  Mat pcZE2L; // Local coarse space (eigen space to local space).
  pcRC = createZE2L(pcAllEigValLoc, pcAllEigVecLoc, gCtx, pcZE2L);
  CHKERRQ(pcRC);
  pcRC = createZE2G(pcAllEigVecLoc, pcZE2L, gCtx);
  CHKERRQ(pcRC);
  auto stop = chrono::high_resolution_clock::now();
  gCtx->lvl2SetupZTimeLoc = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  petscWorld.barrier(); // Synchronisation before creating E (= global distributed built by all processus).

  start = chrono::high_resolution_clock::now();
  pcRC = createEEig(gCtx, pcA);
  CHKERRQ(pcRC);
  stop = chrono::high_resolution_clock::now();
  gCtx->lvl2SetupETimeLoc = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  petscWorld.barrier(); // Synchronisation after creating E (= global distributed built by all processus).

  // Clean.

  pcRC = EPSDestroy(&scEPSLoc);
  CHKERRQ(pcRC);
  pcRC = MatDestroy(&pcDADirDLoc);
  CHKERRQ(pcRC);

  return 0;
}

string getConvergedReason(KSPConvergedReason const & pcReason) {
  string reason = "";

  /* converged */
  if (pcReason == KSP_CONVERGED_ITERATING      ) reason = "KSP_CONVERGED_ITERATING";
  if (pcReason == KSP_CONVERGED_RTOL_NORMAL    ) reason = "KSP_CONVERGED_RTOL_NORMAL";
  if (pcReason == KSP_CONVERGED_ATOL_NORMAL    ) reason = "KSP_CONVERGED_ATOL_NORMAL";
  if (pcReason == KSP_CONVERGED_RTOL           ) reason = "KSP_CONVERGED_RTOL";
  if (pcReason == KSP_CONVERGED_ATOL           ) reason = "KSP_CONVERGED_ATOL";
  if (pcReason == KSP_CONVERGED_ITS            ) reason = "KSP_CONVERGED_ITS";
  if (pcReason == KSP_CONVERGED_CG_NEG_CURVE   ) reason = "KSP_CONVERGED_CG_NEG_CURVE";
  if (pcReason == KSP_CONVERGED_CG_CONSTRAINED ) reason = "KSP_CONVERGED_CG_CONSTRAINED";
  if (pcReason == KSP_CONVERGED_STEP_LENGTH    ) reason = "KSP_CONVERGED_STEP_LENGTH";
  if (pcReason == KSP_CONVERGED_HAPPY_BREAKDOWN) reason = "KSP_CONVERGED_HAPPY_BREAKDOWN";
  /* diverged */
  if (pcReason == KSP_DIVERGED_NULL            ) reason = "KSP_DIVERGED_NULL";
  if (pcReason == KSP_DIVERGED_ITS             ) reason = "KSP_DIVERGED_ITS";
  if (pcReason == KSP_DIVERGED_DTOL            ) reason = "KSP_DIVERGED_DTOL";
  if (pcReason == KSP_DIVERGED_BREAKDOWN       ) reason = "KSP_DIVERGED_BREAKDOWN";
  if (pcReason == KSP_DIVERGED_BREAKDOWN_BICG  ) reason = "KSP_DIVERGED_BREAKDOWN_BICG";
  if (pcReason == KSP_DIVERGED_NONSYMMETRIC    ) reason = "KSP_DIVERGED_NONSYMMETRIC";
  if (pcReason == KSP_DIVERGED_INDEFINITE_PC   ) reason = "KSP_DIVERGED_INDEFINITE_PC";
  if (pcReason == KSP_DIVERGED_NANORINF        ) reason = "KSP_DIVERGED_NANORINF";
  if (pcReason == KSP_DIVERGED_INDEFINITE_MAT  ) reason = "KSP_DIVERGED_INDEFINITE_MAT";
  if (pcReason == KSP_DIVERGED_PCSETUP_FAILED  ) reason = "KSP_DIVERGED_PCSETUP_FAILED";

  return reason;
}

PetscErrorCode checkKSPSolve(KSP const & pcKSP, geneoContext const * const gCtx,
                             bool const & localSolve, string const & info) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Get reason.

  KSPConvergedReason pcReason;
  PetscErrorCode pcRC = KSPGetConvergedReason(pcKSP, &pcReason);
  CHKERRQ(pcRC);
  string reason = getConvergedReason(pcReason);

  // Debug on demand or on failure.

  if (pcReason < 0 || gCtx->debug >= 2) {
    PetscViewer pcView;
    if (localSolve) {
      string debugKSPFile = gCtx->debugFile + ".apply." + info;
      pcRC = createViewer(false, false, PETSC_COMM_SELF, debugKSPFile, pcView);
      CHKERRQ(pcRC);
    }
    else {
      string debugKSPFile = "debug.apply." + info;
      pcRC = createViewer(false, false, PETSC_COMM_WORLD, debugKSPFile, pcView);
      CHKERRQ(pcRC);
    }
    pcRC = KSPView(pcKSP, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  if (pcReason < 0) {
    stringstream msg; msg << "GenEO - solve KO: " << info << " (" << reason << ")";
    SETERRABT(msg.str().c_str());
  }
  return 0;
}

PetscErrorCode applyQ(geneoContext * const gCtx, Vec const & pcX, Vec & pcQX, string const & info, // Apply Q = Z*E^-1*Zt.
                      double * const applyZtTimeLoc, double * const applyEinvTimeLoc, double * const applyZTimeLoc) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Debug on demand.

  PetscErrorCode pcRC;
  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = "debug." + info + ".applyQ.X";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(pcX, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  // Offload on demand.

  if (gCtx->offload) { // Collective scatter.
    pcRC = VecScatterBegin(gCtx->pcScatCtxOff, pcX, gCtx->pcXOff, INSERT_VALUES, SCATTER_FORWARD); // Scatter.
    CHKERRQ(pcRC);
    pcRC = VecScatterEnd(gCtx->pcScatCtxOff, pcX, gCtx->pcXOff, INSERT_VALUES, SCATTER_FORWARD); // Scatter.
    CHKERRQ(pcRC);
  }

  // Apply Zt to X (global space to eigen space).

  auto start = chrono::high_resolution_clock::now();
  if (gCtx->offload) {
    boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
    if (petscWorld.rank() == 0) {
      if (!gCtx->pcZE2GOff) SETERRABT("GenEO preconditioner: offload Z KO");
      pcRC = MatMultTranspose(*(gCtx->pcZE2GOff), gCtx->pcXOff, gCtx->pcYEigOff);
      CHKERRQ(pcRC);
    }
  }
  else {
    pcRC = MatMultTranspose(gCtx->pcZE2G, pcX, gCtx->pcYEig);
    CHKERRQ(pcRC);
  }
  auto stop = chrono::high_resolution_clock::now();
  if (applyZtTimeLoc) *applyZtTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  // Direct local coarse-projected solve (eigen space to eigen space): apply E^⁻1.

  start = chrono::high_resolution_clock::now();
  if (gCtx->offload) {
    boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
    if (petscWorld.rank() == 0) {
      pcRC = KSPSolve(gCtx->pcKSPL2Off, gCtx->pcYEigOff, gCtx->pcYEigOff);
      CHKERRQ(pcRC);
      pcRC = checkKSPSolve(gCtx->pcKSPL2Off, gCtx, true, "dcs2");
      CHKERRQ(pcRC);
    }
  }
  else {
    pcRC = KSPSolve(gCtx->pcKSPL2, gCtx->pcYEig, gCtx->pcYEig);
    CHKERRQ(pcRC);
    pcRC = checkKSPSolve(gCtx->pcKSPL2, gCtx, false, "dcs2");
    CHKERRQ(pcRC);
  }
  stop = chrono::high_resolution_clock::now();
  if (applyEinvTimeLoc) *applyEinvTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  // Apply Z (eigen space to global space) to get X = Q*X with Q = Z*E^-1*Zt.

  start = chrono::high_resolution_clock::now();
  if (gCtx->offload) {
    boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
    if (petscWorld.rank() == 0) {
      if (!gCtx->pcZE2GOff) SETERRABT("GenEO preconditioner: offload Z KO");
      pcRC = MatMult(*(gCtx->pcZE2GOff), gCtx->pcYEigOff, gCtx->pcXOff);
      CHKERRQ(pcRC);
    }
  }
  else {
    pcRC = MatMult(gCtx->pcZE2G, gCtx->pcYEig, pcQX);
    CHKERRQ(pcRC);
  }
  stop = chrono::high_resolution_clock::now();
  if (applyZTimeLoc) *applyZTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  // Offload on demand.

  if (gCtx->offload) { // Collective gather.
    pcRC = VecScatterBegin(gCtx->pcScatCtxOff, gCtx->pcXOff, pcQX, INSERT_VALUES, SCATTER_REVERSE); // Gather.
    CHKERRQ(pcRC);
    pcRC = VecScatterEnd(gCtx->pcScatCtxOff, gCtx->pcXOff, pcQX, INSERT_VALUES, SCATTER_REVERSE); // Gather.
    CHKERRQ(pcRC);
  }

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = "debug." + info + ".applyQ.QX";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(pcQX, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  return 0;
}

PetscErrorCode setUpLevel2(geneoContext * const gCtx,
                           Mat const & pcANeuLoc, Mat const * const pcADirLoc, Mat const & pcARobLoc,
                           Mat const & pcA) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");
  if (!pcADirLoc) SETERRABT("GenEO preconditioner without dirichlet matrix");

  // Use GenEO to build the coarse space.

  PetscErrorCode pcRC = buildCoarseSpaceWithGenEO(pcANeuLoc, pcADirLoc, pcARobLoc, gCtx, pcA);
  CHKERRQ(pcRC);

  // Create local vector(s) and scatter-gather context if needed.

  if (gCtx->offload) {
    // Create local vectors.

    boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
    if (petscWorld.rank() == 0) {
      if (!gCtx->pcZE2GOff) SETERRABT("GenEO preconditioner: offload Z KO");
      pcRC = MatCreateVecs(*(gCtx->pcZE2GOff), &(gCtx->pcYEigOff), NULL);
      CHKERRQ(pcRC);
      pcRC = VecCreateSeq(PETSC_COMM_SELF, gCtx->nbDOF, &(gCtx->pcXOff));
      CHKERRQ(pcRC);
    }

    // Create scatter-gather context.

    pcRC = VecScatterCreateToZero(gCtx->pcX, &(gCtx->pcScatCtxOff), &(gCtx->pcXOff));
    CHKERRQ(pcRC);

    // Clean distributed Z and E (we'll use the offloaded sequential version on the master proc.).

    pcRC = MatDestroy(&(gCtx->pcZE2G));
    CHKERRQ(pcRC);
    gCtx->pcZE2G = NULL;
    pcRC = MatDestroy(&(gCtx->pcEEig));
    CHKERRQ(pcRC);
    gCtx->pcEEig = NULL;
  }
  else {
    // Create local vectors.

    pcRC = MatCreateVecs(gCtx->pcZE2G, &(gCtx->pcYEig), NULL);
    CHKERRQ(pcRC);
  }

  // Modify initial guess on demand (efficient hybrid).

  // If we didn't provide initial guess and RHS.
  if (!gCtx->pcX0) {
    pcRC = MatCreateVecs(pcA, &gCtx->pcX0, NULL);
    CHKERRQ(pcRC);
  }
  if (!gCtx->pcB) {
    pcRC = MatCreateVecs(pcA, NULL, &gCtx->pcB);
    CHKERRQ(pcRC);
  }
  if (gCtx->effHybrid) {
    pcRC = applyQ(gCtx, gCtx->pcB, gCtx->pcX0, "setup.initialGuess", NULL, NULL, NULL); // X0 = Q*B.
    CHKERRQ(pcRC);
  }
  else {
    pcRC = VecSet(gCtx->pcX0, 0.);
    CHKERRQ(pcRC);
  }

  return 0;
}

PetscErrorCode createRobinMatrix(geneoContext * const gCtx,
                                 Mat const & pcANeuLoc, Mat const * const pcADirLoc, Mat & pcARobLoc) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");
  if (!pcADirLoc) SETERRABT("GenEO preconditioner without dirichlet matrix");
  if (!gCtx->dofIdxMultLoc) SETERRABT("GenEO preconditioner without DOF multiplicity");

  pcARobLoc = NULL;
  if (!gCtx->lvl1ORAS) return 0; // Nothing to do.

  // Initialize Robin matrix with Dirichlet matrix.

  PetscErrorCode pcRC = MatDuplicate(*pcADirLoc, MAT_COPY_VALUES, &pcARobLoc);
  CHKERRQ(pcRC);
  double const eps = numeric_limits<double>::epsilon();
  if (fabs(gCtx->optim) <= eps) return 0; // OK, we are done.

  // Find border (where multiplicity is > 1).

  vector<PetscInt> pcBorderDOF;
  for (auto idxMult = gCtx->dofIdxMultLoc->cbegin(); idxMult != gCtx->dofIdxMultLoc->cend(); idxMult++) {
    if (*idxMult > 1) {
      PetscInt dofID = distance(gCtx->dofIdxMultLoc->cbegin(), idxMult);
      pcBorderDOF.push_back(dofID);
    }
  }
  PetscInt pcNbBorderDOF = pcBorderDOF.size();
  if (pcNbBorderDOF == 0) return 0; // OK, we are done.

  // Extract border values from Neumann matrix.

  vector<PetscScalar> pcBorderNeuVal; // Neumann values at the border.
  pcBorderNeuVal.assign(pcNbBorderDOF * pcNbBorderDOF, 0.); // Allocate space to store values.
  pcRC = MatGetValues(pcANeuLoc, pcNbBorderDOF, pcBorderDOF.data(), pcNbBorderDOF, pcBorderDOF.data(), pcBorderNeuVal.data());
  CHKERRQ(pcRC);

  Mat pcSubANeuLoc; // Sub matrix extracted from Neumann matrix, but, of the same dimension than Neumann matrix.
  pcRC = MatCreateSeqAIJ(PETSC_COMM_SELF, gCtx->nbDOFLoc, gCtx->nbDOFLoc, pcNbBorderDOF, NULL, &pcSubANeuLoc);
  CHKERRQ(pcRC);
  pcRC = MatSetValues(pcSubANeuLoc, pcNbBorderDOF, pcBorderDOF.data(), pcNbBorderDOF, pcBorderDOF.data(),
                      pcBorderNeuVal.data(), INSERT_VALUES);
  CHKERRQ(pcRC);
  pcRC = MatAssemblyBegin(pcSubANeuLoc, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);
  pcRC = MatAssemblyEnd  (pcSubANeuLoc, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);

  // Robin = Dirichlet + optim * Neumann.

  pcRC = MatAXPY(pcARobLoc, gCtx->optim, pcSubANeuLoc, DIFFERENT_NONZERO_PATTERN);
  CHKERRQ(pcRC);

  // Clean.

  pcRC = MatDestroy(&pcSubANeuLoc);
  CHKERRQ(pcRC);

  return 0;
}

static PetscErrorCode setUpGenEOPC(PC pcPC) {
  // Get the context.

  if (!pcPC) SETERRABT("GenEO preconditioner is invalid");
  geneoContext * gCtx = (geneoContext *) pcPC->data;
  if (!gCtx) SETERRABT("GenEO preconditioner without context");
  MatType pcMatType = NULL;
  PetscErrorCode pcRC = MatGetType(gCtx->pcA, &pcMatType);
  CHKERRQ(pcRC);
  if (strcmp(MATIS, pcMatType) != 0) SETERRABT("GenEO preconditioner needs the A matrix to be of MATIS type");

  // Ensure slepc is initialised
  PetscBool init;
  pcRC = SlepcInitialized(&init);
  CHKERRQ(pcRC);
  if (!init) {
    pcRC = SlepcInitialize(NULL, NULL, NULL, NULL);
    CHKERRQ(pcRC);
  }
  // Dirichlet matrix: local matrix (extracted by domain) from A after assembly.

  Mat pcA; // Get A as a MatMPI matrix (not a MatIS).
  pcRC = MatConvert(gCtx->pcA, MATAIJ, MAT_INITIAL_MATRIX, &pcA); // Assemble local parts of A.
  CHKERRQ(pcRC);
  Mat * pcADirLoc = NULL; // Dirichlet matrix.
  pcRC = MatCreateSubMatrices(pcA, 1, &(gCtx->pcIS), &(gCtx->pcIS), MAT_INITIAL_MATRIX, &pcADirLoc);
  CHKERRQ(pcRC); // Use MatCreateSubMatrices to get sequential matrix (MatCreateSubMatrix would return distributed matrix).

  // Get Neumann matrix: local matrix (extracted by domain) from A before assembly.

  Mat pcANeuLoc;
  pcRC = MatISGetLocalMat(gCtx->pcA, &pcANeuLoc);
  CHKERRQ(pcRC);

  // Get Robin matrix: local matrix.

  Mat pcARobLoc = NULL;
  pcRC = createRobinMatrix(gCtx, pcANeuLoc, pcADirLoc, pcARobLoc);
  CHKERRQ(pcRC);

  // Set up preconditioner.

  bool needPartitionOfUnity = (gCtx->lvl2 || gCtx->lvl1RAS) ? true : false;
  if (needPartitionOfUnity) {
    pcRC = createPartitionOfUnity(gCtx); // (1.25) from R1.
    CHKERRQ(pcRC);
  }

  pcRC = setUpLevel1(gCtx, pcADirLoc, pcARobLoc);
  CHKERRQ(pcRC);
  if (gCtx->lvl2) {
    pcRC = setUpLevel2(gCtx, pcANeuLoc, pcADirLoc, pcARobLoc, pcA);
    CHKERRQ(pcRC);
  }

  // Check on demand.

  if (gCtx->check) {
    pcRC = checkSPD(false, pcA, "check", "A");
    CHKERRQ(pcRC);
  }

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = "debug.setup.A.MatMPI";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = MatView(pcA, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);

    if (pcADirLoc) {
      string debugFile = gCtx->debugFile + ".setup.ADir";
      pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_SELF, debugFile, pcView);
      CHKERRQ(pcRC);
      pcRC = MatView(*pcADirLoc, pcView);
      CHKERRQ(pcRC);
      pcRC = PetscViewerDestroy(&pcView);
      CHKERRQ(pcRC);
    }

    debugFile = gCtx->debugFile + ".setup.ANeu";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_SELF, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = MatView(pcANeuLoc, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);

    if (pcARobLoc) {
      debugFile = gCtx->debugFile + ".setup.ARob";
      pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_SELF, debugFile, pcView);
      CHKERRQ(pcRC);
      pcRC = MatView(pcARobLoc, pcView);
      CHKERRQ(pcRC);
      pcRC = PetscViewerDestroy(&pcView);
      CHKERRQ(pcRC);
    }

    if (!gCtx->dofIdxMultLoc) SETERRABT("GenEO preconditioner: shell preconditioner without DOF multiplicities");
    ofstream debugDOF(gCtx->debugFile + ".input.dof.log");
    PetscInt pcNbIdxLoc = 0;
    pcRC = ISLocalToGlobalMappingGetSize(gCtx->pcMap, &pcNbIdxLoc);
    CHKERRQ(pcRC);
    const PetscInt * pcMapIndices = NULL;
    pcRC = ISLocalToGlobalMappingGetIndices(gCtx->pcMap, &pcMapIndices);
    CHKERRQ(pcRC);
    for (PetscInt pcIdxLoc = 0; pcMapIndices && pcIdxLoc < pcNbIdxLoc; pcIdxLoc++) {
      debugDOF << "DOF " << pcIdxLoc << ":";
      debugDOF << " global index " << pcMapIndices[pcIdxLoc];
      debugDOF << ", multiplicity " << (*(gCtx->dofIdxMultLoc))[pcIdxLoc];
      debugDOF << endl;
    }

    if (!gCtx->intersectLoc) SETERRABT("GenEO preconditioner: shell preconditioner without intersection");
    ofstream debugInt(gCtx->debugFile + ".input.intersect.log");
    boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
    unsigned int p = petscWorld.rank();
    for (unsigned int q = 0; pcMapIndices && q < gCtx->intersectLoc->size(); q++) { // Loop over all partitions.
      if (p == q) continue; // Skip self intersection !

      vector<unsigned int> const & intersectPQLoc = (*(gCtx->intersectLoc))[q];
      for (auto idxLoc = intersectPQLoc.cbegin(); idxLoc != intersectPQLoc.cend(); idxLoc++) {
        PetscInt pcIdxGlob = pcMapIndices[*idxLoc];
        debugInt << "domains " << p << " and " << q;
        debugInt << " intersect in global index " << pcIdxGlob << " (local index " << *idxLoc << ")";
        debugInt << " with multiplicity " << (*(gCtx->dofIdxMultLoc))[*idxLoc];
        debugInt << endl;
      }
    }
    pcRC = ISLocalToGlobalMappingRestoreIndices(gCtx->pcMap, &pcMapIndices);
    CHKERRQ(pcRC);

    if (gCtx->pcIS) {
      string debugFile = "debug.input.IS";
      pcRC = createViewer(false, false, PETSC_COMM_WORLD, debugFile, pcView);
      CHKERRQ(pcRC);
      pcRC = ISView(gCtx->pcIS, pcView);
      CHKERRQ(pcRC);
      pcRC = PetscViewerDestroy(&pcView);
      CHKERRQ(pcRC);
    }
  }

  // Clean.

  if (pcARobLoc) {
    pcRC = MatDestroy(&pcARobLoc);
    CHKERRQ(pcRC);
  }
  pcRC = MatDestroyMatrices(1, &pcADirLoc);
  CHKERRQ(pcRC);
  pcRC = MatDestroy(&pcA);
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode applyLevel1Scatter(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Restrict the global X to the local domain: use INSERT_VALUES as we want to retrieve local copies.

  PetscErrorCode pcRC = VecScatterBegin(gCtx->pcScatCtx, gCtx->pcX, gCtx->pcXLoc, INSERT_VALUES, SCATTER_FORWARD); // Scatter.
  CHKERRQ(pcRC);
  pcRC = VecScatterEnd(gCtx->pcScatCtx, gCtx->pcX, gCtx->pcXLoc, INSERT_VALUES, SCATTER_FORWARD); // Scatter.
  CHKERRQ(pcRC);

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = gCtx->debugFile + ".apply.L1.scatter.X";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_SELF, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(gCtx->pcXLoc, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  return 0;
}

PetscErrorCode applyLevel1Gather(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Reset X: the old solution must be replaced by the new one (gather with ADD_VALUES will add old and new solutions).

  PetscErrorCode pcRC = VecSet(gCtx->pcX, 0.);
  CHKERRQ(pcRC);

  // Global assembly of local solutions: use ADD_VALUES to sum possibly overlapping index sets.

  pcRC = VecScatterBegin(gCtx->pcScatCtx, gCtx->pcXLoc, gCtx->pcX, ADD_VALUES, SCATTER_REVERSE); // Gather.
  CHKERRQ(pcRC);
  pcRC = VecScatterEnd(gCtx->pcScatCtx, gCtx->pcXLoc, gCtx->pcX, ADD_VALUES, SCATTER_REVERSE); // Gather.
  CHKERRQ(pcRC);

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = "debug.apply.L1.gather.X";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(gCtx->pcX, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  return 0;
}

PetscErrorCode projectOnFineSpace(geneoContext * const gCtx, Vec const * const pcQX) { // Apply I-P, or, I-Pt.
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Save X into XOld.

  if (!gCtx->pcXOld) {
    PetscErrorCode pcRC = VecDuplicate(gCtx->pcX, &(gCtx->pcXOld)); // Create but do not fill.
    CHKERRQ(pcRC);
  }
  PetscErrorCode pcRC = VecCopy(gCtx->pcX, gCtx->pcXOld); // XOld = X.
  CHKERRQ(pcRC);

  // Debug on demand.

  string operation = (pcQX) ? "I-Pt" : "I-P";
  PetscViewer pcView;
  if (gCtx->debug >= 2) {
    string debugFile = "debug.apply.L1.projFS." + operation + ".old.X";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(gCtx->pcX, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  // Apply Pt or P.

  if (pcQX) { // Apply Pt to X with P = QA. Need to apply Pt = AQ: QX is known from level 2, it remains to apply A.
    pcRC = MatMult(gCtx->pcA, *pcQX, gCtx->pcX); // X = A*Q*XOld = Pt*XOld.
    CHKERRQ(pcRC);
  }
  else { // Apply P to X with P = QA.
    pcRC = MatMult(gCtx->pcA, gCtx->pcXOld, gCtx->pcX); // X = A*XOld.
    CHKERRQ(pcRC);
    pcRC = applyQ(gCtx, gCtx->pcX, gCtx->pcX, "apply.L1.projFS." + operation + ".old.applyP", // X = Q*A*XOld = P*XOld.
                  &(gCtx->lvl1ApplyPrjFSZtTimeLoc), &(gCtx->lvl1ApplyPrjFSEinvTimeLoc), &(gCtx->lvl1ApplyPrjFSZTimeLoc));
    CHKERRQ(pcRC);
  }

  // Substract XOld.

  pcRC = VecAXPBY(gCtx->pcX, 1., -1., gCtx->pcXOld); // X = XOld - X.
  CHKERRQ(pcRC);

  // Debug on demand.

  if (gCtx->debug >= 2) {
    string debugFile = "debug.apply.L1.projFS." + operation + ".new.X";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(gCtx->pcX, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  return 0;
}

PetscErrorCode applyLevel1(geneoContext * const gCtx, Vec const & pcQX, Vec const & pcX) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // PETSc forbids to modify X. Create a copy of X that can be modified.

  PetscErrorCode pcRC = VecCopy(pcX, gCtx->pcX); // X = X.
  CHKERRQ(pcRC);

  // Apply projection on the fine space (= complementary of the coarse space).

  if (gCtx->hybrid && !gCtx->effHybrid) {
    auto start = chrono::high_resolution_clock::now();
    pcRC = projectOnFineSpace(gCtx, &pcQX); // Apply (I-Pt) in 2nd term of (7.53) from R1.
    CHKERRQ(pcRC);
    auto stop = chrono::high_resolution_clock::now();
    gCtx->lvl1ApplyPrjFSTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
  }

  // Scatter.

  auto start = chrono::high_resolution_clock::now();
  pcRC = applyLevel1Scatter(gCtx);
  CHKERRQ(pcRC);
  auto stop = chrono::high_resolution_clock::now();
  gCtx->lvl1ApplyScatterTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  // Direct local dirichlet solve.

  start = chrono::high_resolution_clock::now();
  if (gCtx->lvl1RAS) {
    pcRC = VecPointwiseMult(gCtx->pcXLoc, gCtx->pcDLoc, gCtx->pcXLoc); // (1.29) from R1.
    CHKERRQ(pcRC);
  }
  pcRC = KSPSolve(gCtx->pcKSPL1Loc, gCtx->pcXLoc, gCtx->pcXLoc); // X = M^-1*X. Central part of 2nd term of (7.53) from R1.
  CHKERRQ(pcRC);
  pcRC = checkKSPSolve(gCtx->pcKSPL1Loc, gCtx, true, "dls1");
  CHKERRQ(pcRC);
  if (gCtx->lvl1SRAS) {
    pcRC = VecPointwiseMult(gCtx->pcXLoc, gCtx->pcDLoc, gCtx->pcXLoc); // (1.29) from R1.
    CHKERRQ(pcRC);
  }
  stop = chrono::high_resolution_clock::now();
  gCtx->lvl1ApplyMinvTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  // Debug on demand.

  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = gCtx->debugFile + ".apply.L1.X";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_SELF, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(gCtx->pcXLoc, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  // Gather.

  start = chrono::high_resolution_clock::now();
  pcRC = applyLevel1Gather(gCtx);
  CHKERRQ(pcRC);
  stop = chrono::high_resolution_clock::now();
  gCtx->lvl1ApplyGatherTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  // Apply projection on the fine space (= complementary of the coarse space).

  if (gCtx->hybrid) {
    auto start = chrono::high_resolution_clock::now();
    pcRC = projectOnFineSpace(gCtx, NULL); // Apply (I-P) in 2nd term of (7.53) from R1.
    CHKERRQ(pcRC);
    auto stop = chrono::high_resolution_clock::now();
    gCtx->lvl1ApplyPrjFSTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
  }

  return 0;
}

PetscErrorCode applyLevel2(geneoContext * const gCtx, Vec const & pcX, Vec & pcY) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  if (gCtx->effHybrid) return 0; // We are done: level 2 is not needed.
  PetscErrorCode pcRC = applyQ(gCtx, pcX, pcY, "apply.L2", // Y = Q*X.
                               &(gCtx->lvl2ApplyZtTimeLoc), &(gCtx->lvl2ApplyEinvTimeLoc), &(gCtx->lvl2ApplyZTimeLoc));
  CHKERRQ(pcRC);

  return 0;
}

static PetscErrorCode applyGenEOPC(PC pcPC, Vec pcX, Vec pcY) {
  // Get the context.

  if (!pcPC) SETERRABT("GenEO preconditioner is invalid");
  geneoContext * gCtx = (geneoContext *) pcPC->data;
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Debug on demand.

  PetscErrorCode pcRC;
  if (gCtx->debug >= 2) {
    PetscViewer pcView;
    string debugFile = "debug.apply.input.X";
    pcRC = createViewer(gCtx->debugBin, gCtx->debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(pcX, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  // First, handle level 2 with Y (Y = Q*X): 1st term of (7.53) from R1. X is untouched.

  pcRC = VecSet(pcY, 0.); // Reset.
  CHKERRQ(pcRC);
  if (gCtx->lvl2) {
    auto start = chrono::high_resolution_clock::now();
    pcRC = applyLevel2(gCtx, pcX, pcY); // Y holds contribution of level 2 (Y = Q*X).
    CHKERRQ(pcRC);
    auto stop = chrono::high_resolution_clock::now();
    gCtx->lvl2ApplyTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
  }

  // Now, handle level 1 with gCtx->X: 2nd term of (7.53) from R1. X is untouched, Y (= Q*X) may be reused.

  auto start = chrono::high_resolution_clock::now();
  pcRC = applyLevel1(gCtx, pcY, pcX); // gCtx->X holds contribution of level 1.
  CHKERRQ(pcRC);
  auto stop = chrono::high_resolution_clock::now();
  gCtx->lvl1ApplyTimeLoc += chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  // Then, add contributions of levels 1 (gCtx->X) and 2 (Y).

  pcRC = VecAXPY(pcY, 1., gCtx->pcX); // Y += X.
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode destroyLevel1(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  PetscErrorCode pcRC;
  if (gCtx->pcX) {
    pcRC = VecDestroy(&(gCtx->pcX));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcXLoc) {
    pcRC = VecDestroy(&(gCtx->pcXLoc));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcXOld) {
    pcRC = VecDestroy(&(gCtx->pcXOld));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcScatCtx) {
    pcRC = VecScatterDestroy(&(gCtx->pcScatCtx));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcKSPL1Loc) {
    pcRC = KSPDestroy(&(gCtx->pcKSPL1Loc));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcDLoc) {
    pcRC = VecDestroy(&(gCtx->pcDLoc));
    CHKERRQ(pcRC);
  }

  return 0;
}

PetscErrorCode destroyLevel2(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  PetscErrorCode pcRC;
  if (gCtx->pcZE2G) {
    pcRC = MatDestroy(&(gCtx->pcZE2G));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcZE2GOff) {
    pcRC = MatDestroyMatrices(1, &(gCtx->pcZE2GOff));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcEEig) {
    pcRC = MatDestroy(&(gCtx->pcEEig));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcEEigOff) {
    pcRC = MatDestroyMatrices(1, &(gCtx->pcEEigOff));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcKSPL2) {
    pcRC = KSPDestroy(&(gCtx->pcKSPL2));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcKSPL2Off) {
    pcRC = KSPDestroy(&(gCtx->pcKSPL2Off));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcScatCtxOff) {
    pcRC = VecScatterDestroy(&(gCtx->pcScatCtxOff));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcXOff) {
    pcRC = VecDestroy(&(gCtx->pcXOff));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcYEig) {
    pcRC = VecDestroy(&(gCtx->pcYEig));
    CHKERRQ(pcRC);
  }
  if (gCtx->pcYEigOff) {
    pcRC = VecDestroy(&(gCtx->pcYEigOff));
    CHKERRQ(pcRC);
  }

  return 0;
}

static PetscErrorCode destroyGenEOPC(PC pcPC) {
  // Get the context.
  PetscErrorCode pcRC;
  if (!pcPC) SETERRABT("GenEO preconditioner is invalid");
  geneoContext * gCtx = (geneoContext *) pcPC->data;
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Debug on demand.

  if (gCtx->debug >= 1) {
    ofstream debug(gCtx->debugFile + ".timing.log");
    debug << "lvl1SetupMinvTimeLoc      " << gCtx->lvl1SetupMinvTimeLoc      << " ms" << endl;
    debug << "lvl1ApplyTimeLoc          " << gCtx->lvl1ApplyTimeLoc          << " ms" << endl;
    debug << "lvl1ApplyScatterTimeLoc   " << gCtx->lvl1ApplyScatterTimeLoc   << " ms" << endl;
    debug << "lvl1ApplyMinvTimeLoc      " << gCtx->lvl1ApplyMinvTimeLoc      << " ms" << endl;
    debug << "lvl1ApplyGatherTimeLoc    " << gCtx->lvl1ApplyGatherTimeLoc    << " ms" << endl;
    debug << "lvl1ApplyPrjFSTimeLoc     " << gCtx->lvl1ApplyPrjFSTimeLoc     << " ms" << endl;
    debug << "lvl1ApplyPrjFSZtTimeLoc   " << gCtx->lvl1ApplyPrjFSZtTimeLoc   << " ms" << endl;
    debug << "lvl1ApplyPrjFSEinvTimeLoc " << gCtx->lvl1ApplyPrjFSEinvTimeLoc << " ms" << endl;
    debug << "lvl1ApplyPrjFSZTimeLoc    " << gCtx->lvl1ApplyPrjFSZTimeLoc    << " ms" << endl;
    debug << endl;
    debug << "lvl2SetupTauLocTimeLoc    " << gCtx->lvl2SetupTauLocTimeLoc    << " ms" << endl;
    debug << "lvl2SetupTauSylTimeLoc    " << gCtx->lvl2SetupTauSylTimeLoc    << " ms" << endl;
    debug << "lvl2SetupTauEigTimeLoc    " << gCtx->lvl2SetupTauEigTimeLoc    << " ms" << endl;
    debug << "lvl2SetupGammaLocTimeLoc  " << gCtx->lvl2SetupGammaLocTimeLoc  << " ms" << endl;
    debug << "lvl2SetupGammaSylTimeLoc  " << gCtx->lvl2SetupGammaSylTimeLoc  << " ms" << endl;
    debug << "lvl2SetupGammaEigTimeLoc  " << gCtx->lvl2SetupGammaEigTimeLoc  << " ms" << endl;
    debug << "lvl2SetupSylTimeLoc       " << gCtx->lvl2SetupSylTimeLoc       << " ms" << endl;
    debug << "lvl2SetupEigTimeLoc       " << gCtx->lvl2SetupEigTimeLoc       << " ms" << endl;
    debug << "lvl2SetupZTimeLoc         " << gCtx->lvl2SetupZTimeLoc         << " ms" << endl;
    debug << "lvl2SetupETimeLoc         " << gCtx->lvl2SetupETimeLoc         << " ms" << endl;
    debug << "lvl2ApplyTimeLoc          " << gCtx->lvl2ApplyTimeLoc          << " ms" << endl;
    debug << "lvl2ApplyZtTimeLoc        " << gCtx->lvl2ApplyZtTimeLoc        << " ms" << endl;
    debug << "lvl2ApplyEinvTimeLoc      " << gCtx->lvl2ApplyEinvTimeLoc      << " ms" << endl;
    debug << "lvl2ApplyZTimeLoc         " << gCtx->lvl2ApplyZTimeLoc         << " ms" << endl;
  }

  // Clean.

  gCtx->nbDOF = 0;
  gCtx->nbDOFLoc = 0;
  gCtx->pcMap = NULL; // Do not destroy (creation has been done out of context).
  gCtx->pcA = NULL; // Do not destroy (creation has been done out of context).
  pcRC = VecDestroy(&gCtx->pcB); CHKERRQ(pcRC);
  pcRC = VecDestroy(&gCtx->pcX0); CHKERRQ(pcRC);
  if (gCtx->pcIS) {
    pcRC = ISDestroy(&(gCtx->pcIS));
    CHKERRQ(pcRC);
  }
  gCtx->dofIdxMultLoc = NULL; // Do not destroy (creation has been done out of context).
  gCtx->intersectLoc = NULL; // Do not destroy (creation has been done out of context).

  pcRC = destroyLevel1(gCtx);
  CHKERRQ(pcRC);
  if (gCtx->lvl2) {
    pcRC = destroyLevel2(gCtx);
    CHKERRQ(pcRC);
  }

  delete gCtx; gCtx = NULL;
  pcPC->data = gCtx;

  return 0;
}

PetscErrorCode buildGenEOName(geneoContext * const gCtx) {
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Build preconditioner name.

  string name = "geneo";
  if (gCtx->lvl2 == 0) name += "0";
  if (gCtx->lvl2 == 1) name += "1";
  if (gCtx->lvl2 == 2) name += "2";
  if (gCtx->hybrid) {
    if (gCtx->effHybrid) name += "E";
    else                 name += "H";
  }
  string lvl1;
  if (gCtx->lvl1ASM)                    lvl1 = "ASM";
  if (gCtx->lvl1RAS)                    lvl1 = "RAS";
  if (gCtx->lvl1SRAS)                   lvl1 = "SRAS";
  if (gCtx->lvl1ORAS)                   lvl1 = "ORAS";
  if (gCtx->lvl1SRAS && gCtx->lvl1ORAS) lvl1 = "SORAS";
  name += lvl1;
  gCtx->name = name;

  return 0;
}

/*
 * usageGenEO: usage of GenEO.
 *   - petscPrintf: do or don't PetscPrintf (PETSC_COMM_WORLD).
 */
string usageGenEO(bool const petscPrintf) {
  stringstream msg;
  msg << endl;
  msg << "usage: implementation of GenEO (Domain Decomposition Method) with PETSc and SLEPc" << endl;
  msg << "" << endl;
  msg << "  -geneo_lvl L1,L2 preconditioner with 2 levels L1 and L2" << endl;
  msg << "                   L1 =   ASM = Additive Schwarz Method" << endl;
  msg << "                   L1 =   RAS = Restricted Additive Schwarz method" << endl;
  msg << "                   L1 =  SRAS = Symmetrized Restricted Additive Schwarz method" << endl;
  msg << "                   L1 =  ORAS = Optimized Restricted Additive Schwarz method" << endl;
  msg << "                   L1 = SORAS = Symmetrized Optimized Restricted Additive Schwarz method" << endl;
  msg << "                   L2 =     0 = do not use GenEO (enables to compare 1-level with 2-levels methods)" << endl;
  msg << "                   L2 =     1 =                  GenEO-1 (1st level                    +  2d level [coarse space])" << endl;
  msg << "                   L2 =    H1 =           hybrid GenEO-1 (1st level [proj. fine space] +  2d level [coarse space])" << endl;
  msg << "                   L2 =    E1 = efficient hybrid GenEO-1 (initial guess [coarse space] + 1st level [proj. fine space])" << endl;
  msg << "                   L2 =     2 =                  GenEO-2 (1st level                    +  2d level [coarse space])" << endl;
  msg << "                   L2 =    H2 =           hybrid GenEO-2 (1st level [proj. fine space] +  2d level [coarse space])" << endl;
  msg << "                   L2 =    E2 = efficient hybrid GenEO-2 (initial guess [coarse space] + 1st level [proj. fine space])" << endl;
  msg << "                   you can pass arguments to PETSc / SLEPc solvers at the command line using prefix:" << endl;
  msg << "                     -dls1_ for      direct  local solve (level 1): -dls1_pc_factor_mat_solver_package mumps" << endl;
  msg << "                     -syl2_ for   sylvester  local solve (level 2): -syl2_ksp_view" << endl;
  msg << "                     -els2_ for       eigen  local solve (level 2): -els2_eps_max_it 100 -els2_eps_type arnoldi" << endl;
  msg << "                     -dcs2_ for      direct coarse solve (level 2): -dcs2_pc_factor_mat_solver_package mumps" << endl;
  msg << "                     -ubl2_ for upper bound  local solve (level 2): -ubl2_eps_max_it 100" << endl;
  msg << "                     -chks_ for customizing EPS solver used to check for SPD (--check)" << endl;
  msg << "                     -chkr_ for customizing basis vector used to check for rank (--check)" << endl;
  msg << "                   in case a solver diverges (for instance dcs2_), you may add to the command line:" << endl;
  msg << "                     -dcs2_ksp_view:        to see informations related to the solve" << endl;
  msg << "                     -mat_mumps_icntl_33 1: to ask mumps to compute the determinant (if mumps is used)" << endl;
  msg << "  -geneo_optim A   if an optimized method (ORAS, SORAS) is used as level 1, a robin condition is used" << endl;
  msg << "                   to modify the dirichlet matrix. robin = dirichlet + optim * neumann (optim defaults to 0.)" << endl;
  msg << "  -geneo_tau T     tau threshold (needed for GenEO - defaults to 0.1)" << endl;
  msg << "  -geneo_gamma G   gamma threshold (needed for GenEO - defaults to 10.)" << endl;
  msg << "  -geneo_cst       when using SORAS2 (or HSORAS) with GenEO2, do not allow local variations of tau and gamma" << endl;
  msg << "  -geneo_cut C     maximum number of local eigen vectors used to build Z (defaults to false)" << endl;
  msg << "  -geneo_no_syl    do not use Sylvester's law (inertia) to estimate the number of eigen values used to build Z" << endl;
  msg << "                   in case the eigen solver handles this estimation (krylovschur), you don't need to do it" << endl;
  msg << "  -geneo_offload   offload Z and E at master side (gather inputs, local solve, scatter outputs)" << endl;
  msg << "" << endl;
  msg << "In case a problem occurs, the following options may help:" << endl;
  msg << "" << endl;
  msg << "  -geneo_dbg F,D   create debug files" << endl;
  msg << "                   F = log (ASCII file), bin (binary file) or mat (matlab file)" << endl;
  msg << "                   D = 1: timing and residual" << endl;
  msg << "                   D = 2: global and local informations (timings may not be relevant because of dumps)" << endl;
  msg << "  -geneo_chk F     perform additional checks" << endl;
  msg << "                     - check partition of unity" << endl;
  msg << "                     - check matrices are SPD (with solvers prefixed chks_)" << endl;
  msg << "                     - check R from Z=QR (with solvers prefixed chkr_)" << endl;
  msg << "                   F = log (ASCII file), bin (binary file) or mat (matlab file)" << endl;
  msg << endl;
  if (petscPrintf) PetscPrintf(PETSC_COMM_WORLD, msg.str().c_str());
  return msg.str();
}

static PetscErrorCode setUpGenEOPCFromOptions(PetscOptionItems * PetscOptionsObject, PC pcPC) {
  // Get the context.

  if (!pcPC) SETERRABT("GenEO preconditioner is invalid");
  geneoContext * gCtx = (geneoContext *) pcPC->data;
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Check arguments.

  PetscErrorCode pcRC = PetscOptionsHead(PetscOptionsObject, "GenEO options");
  CHKERRQ(pcRC);

  PetscBool pcHasOpt = PETSC_FALSE;
  PetscInt nbArgs = 2;
  char *args[2];
  pcRC = PetscOptionsStringArray("-geneo_lvl", "GenEO levels", "GenEO levels", args, &nbArgs, &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) {
    if (nbArgs != 2) {usageGenEO(); string msg = "invalid option -geneo_lvl"; SETERRABT(msg.c_str());}

    stringstream gL1(args[0]);
    if      (gL1.str() ==   "ASM") gCtx->lvl1ASM = true; // ASM (with Dirichlet matrix).
    else if (gL1.str() ==   "RAS") gCtx->lvl1RAS = true; // RAS (with Dirichlet matrix).
    else if (gL1.str() ==  "SRAS") gCtx->lvl1RAS = gCtx->lvl1SRAS = true; // Symmetric RAS (symmetry wrt partition of unity).
    else if (gL1.str() ==  "ORAS") gCtx->lvl1RAS = gCtx->lvl1ORAS = true; // Optimised RAS: RAS with Robin matrix (instead of Dirichlet).
    else if (gL1.str() == "SORAS") gCtx->lvl1RAS = gCtx->lvl1SRAS = gCtx->lvl1ORAS = true; // SORAS = SRAS + ORAS
    else {usageGenEO(); string msg = "invalid option -geneo_lvl, unknown " + string(args[0]); SETERRABT(msg.c_str());}
    pcRC = PetscFree(args[0]);
    CHKERRQ(pcRC);

    stringstream gL2(args[1]);
    if      (gL2.str() ==  "0")  gCtx->lvl2 = 0;
    else if (gL2.str() ==  "1")  gCtx->lvl2 = 1;
    else if (gL2.str() == "H1") {gCtx->lvl2 = 1; gCtx->hybrid = true;}
    else if (gL2.str() == "E1") {gCtx->lvl2 = 1; gCtx->hybrid = true; gCtx->effHybrid = true;}
    else if (gL2.str() ==  "2")  gCtx->lvl2 = 2;
    else if (gL2.str() == "H2") {gCtx->lvl2 = 2; gCtx->hybrid = true;}
    else if (gL2.str() == "E2") {gCtx->lvl2 = 2; gCtx->hybrid = true; gCtx->effHybrid = true;}
    else {usageGenEO(); string msg = "invalid option -geneo_lvl, unknown " + string(args[1]); SETERRABT(msg.c_str());}
    pcRC = PetscFree(args[1]);
    CHKERRQ(pcRC);
  }

  pcHasOpt = PETSC_FALSE;
  nbArgs = 1;
  pcRC = PetscOptionsStringArray("-geneo_optim", "GenEO optim parameter (ORAS)", "GenEO optim parameter (ORAS)", args, &nbArgs, &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) {
    if (nbArgs != 1) {usageGenEO(); string msg = "invalid option -geneo_optim"; SETERRABT(msg.c_str());}

    stringstream optim(args[0]);
    optim >> gCtx->optim;
    if (!optim) {usageGenEO(); string msg = "invalid option -geneo_optim, bad " + string(args[0]); SETERRABT(msg.c_str());}
    pcRC = PetscFree(args[0]);
    CHKERRQ(pcRC);
  }

  pcHasOpt = PETSC_FALSE;
  nbArgs = 1;
  pcRC = PetscOptionsStringArray("-geneo_tau", "GenEO tau parameter", "GenEO tau parameter", args, &nbArgs, &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) {
    if (nbArgs != 1) {usageGenEO(); string msg = "invalid option -geneo_tau"; SETERRABT(msg.c_str());}

    stringstream tau(args[0]);
    tau >> gCtx->tau;
    if (!tau) {usageGenEO(); string msg = "invalid option -geneo_tau, bad " + string(args[0]); SETERRABT(msg.c_str());}
    pcRC = PetscFree(args[0]);
    CHKERRQ(pcRC);
  }

  pcHasOpt = PETSC_FALSE;
  nbArgs = 1;
  pcRC = PetscOptionsStringArray("-geneo_gamma", "GenEO gamma parameter", "GenEO gamma parameter", args, &nbArgs, &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) {
    if (nbArgs != 1) {usageGenEO(); string msg = "invalid option -geneo_gamma"; SETERRABT(msg.c_str());}

    stringstream gamma(args[0]);
    gamma >> gCtx->gamma;
    if (!gamma) {usageGenEO(); string msg = "invalid option -geneo_gamma, bad " + string(args[0]); SETERRABT(msg.c_str());}
    pcRC = PetscFree(args[0]);
    CHKERRQ(pcRC);
  }

  pcHasOpt = PETSC_FALSE;
  pcRC = PetscOptionsHasName(NULL, NULL, "-geneo_cst", &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) gCtx->cst = true;

  pcHasOpt = PETSC_FALSE;
  nbArgs = 1;
  pcRC = PetscOptionsStringArray("-geneo_cut", "GenEO eigen value cut-off", "GenEO eigen value cut-off", args, &nbArgs, &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) {
    if (nbArgs != 1) {usageGenEO(); string msg = "invalid option -geneo_cut"; SETERRABT(msg.c_str());}

    stringstream cut(args[0]);
    cut >> gCtx->cut;
    if (!cut) {usageGenEO(); string msg = "invalid option -geneo_cut, bad " + string(args[0]); SETERRABT(msg.c_str());}
    pcRC = PetscFree(args[0]);
    CHKERRQ(pcRC);
  }

  pcHasOpt = PETSC_FALSE;
  pcRC = PetscOptionsHasName(NULL, NULL, "-geneo_no_syl", &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) gCtx->noSyl = true;

  pcHasOpt = PETSC_FALSE;
  pcRC = PetscOptionsHasName(NULL, NULL, "-geneo_offload", &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) gCtx->offload = true;

  pcHasOpt = PETSC_FALSE;
  nbArgs = 2;
  pcRC = PetscOptionsStringArray("-geneo_dbg", "GenEO debug", "GenEO debug", args, &nbArgs, &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) {
    if (nbArgs != 2) {usageGenEO(); string msg = "invalid option -geneo_dbg"; SETERRABT(msg.c_str());}

    stringstream debugFile(args[0]);
    if      (debugFile.str() == "bin") gCtx->debugBin = true;
    else if (debugFile.str() == "mat") gCtx->debugMat = true;
    else if (debugFile.str() != "log") {usageGenEO(); string msg = "invalid option -geneo_dbg, unknown " + string(args[0]); SETERRABT(msg.c_str());}
    pcRC = PetscFree(args[0]);
    CHKERRQ(pcRC);

    stringstream debugLevel(args[1]);
    debugLevel >> gCtx->debug;
    if (!debugLevel) {usageGenEO(); string msg = "invalid option -geneo_dbg, bad " + string(args[1]); SETERRABT(msg.c_str());}
    pcRC = PetscFree(args[1]);
    CHKERRQ(pcRC);
  }

  pcHasOpt = PETSC_FALSE;
  nbArgs = 1;
  pcRC = PetscOptionsStringArray("-geneo_chk", "GenEO checks (SPD, ...)", "GenEO checks (SPD, ...)", args, &nbArgs, &pcHasOpt);
  CHKERRQ(pcRC);
  if (pcHasOpt) {
    if (nbArgs != 1) {usageGenEO(); string msg = "invalid option -geneo_chk"; SETERRABT(msg.c_str());}

    stringstream checkFile(args[0]);
    if      (checkFile.str() == "bin") gCtx->checkBin = true;
    else if (checkFile.str() == "mat") gCtx->checkMat = true;
    else if (checkFile.str() != "log") {usageGenEO(); string msg = "invalid option -geneo_chk, unknown " + string(args[0]); SETERRABT(msg.c_str());}
    pcRC = PetscFree(args[0]);
    CHKERRQ(pcRC);

    gCtx->check = true;
  }

  pcRC = PetscOptionsTail();
  CHKERRQ(pcRC);

  // Check option consistency.

  if ((gCtx->lvl2 >= 1) && gCtx->tau <= 0.) SETERRABT("GenEO preconditioner: tau must be > 0.");
  if ((gCtx->lvl2 >= 1) && gCtx->tau >= 1.) SETERRABT("GenEO preconditioner: tau must be < 1.");
  if ((gCtx->lvl2 >= 2) && gCtx->gamma <= 1.) SETERRABT("GenEO preconditioner: gamma must be > 1.");

  // Check.

  if (gCtx->check) {
    boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
    stringstream size; size << petscWorld.size();
    stringstream rank; rank << setfill('0') << setw(size.str().length()) << petscWorld.rank();
    gCtx->checkFile = "check" + rank.str();
  }

  // Debug.

  if (gCtx->debug) {
    boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
    stringstream size; size << petscWorld.size();
    stringstream rank; rank << setfill('0') << setw(size.str().length()) << petscWorld.rank();
    gCtx->debugFile = "debug" + rank.str();
  }

  // Build GenEO name.

  pcRC = buildGenEOName(gCtx);
  CHKERRQ(pcRC);

  return 0;
}

extern "C" {

PETSC_EXTERN PetscErrorCode PCGenEOSetup(PC pc, IS dofMultiplicities, IS *dofIntersections)
{
     PetscErrorCode ierr;
     Mat            P;
     ISLocalToGlobalMapping rmap, cmap;
     PetscInt       n, m, N, M;
     auto *localDofset = new set<unsigned int>;
     auto *dofIdxMultLoc = new vector<unsigned int>;
     auto *intersectLoc = new vector<vector<unsigned int>>;
     const PetscInt    *dofs;
     PetscMPIInt        size;

     PetscFunctionBegin;
     ierr = PCGetOperators(pc, NULL, &P); CHKERRQ(ierr);
     ierr = MatGetLocalToGlobalMapping(P, &rmap, &cmap); CHKERRQ(ierr);
     if (rmap != cmap) {
          SETERRQ(PETSC_COMM_SELF,  PETSC_ERR_ARG_WRONG, "Row and column LGMaps must match");
     }
     ierr = MatGetSize(P, &N, &M); CHKERRQ(ierr);
     if (N != M) {
          SETERRQ(PetscObjectComm((PetscObject)pc),  PETSC_ERR_ARG_WRONG, "Matrix must be square");
     }
     ierr = ISLocalToGlobalMappingGetIndices(rmap, &dofs); CHKERRQ(ierr);
     ierr = ISLocalToGlobalMappingGetSize(rmap, &n); CHKERRQ(ierr);
     for (int i = 0; i < n; i++ ) {
          localDofset->insert(static_cast<unsigned int>(dofs[i]));
     }
     ierr = ISLocalToGlobalMappingRestoreIndices(rmap, &dofs); CHKERRQ(ierr);

     ierr = ISGetLocalSize(dofMultiplicities, &m); CHKERRQ(ierr);
     if (n != m) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mismatch in dof mult size and local size");
     ierr = ISGetIndices(dofMultiplicities, &dofs); CHKERRQ(ierr);
     dofIdxMultLoc->reserve(n);
     for (int i = 0; i < n; i++) {
          dofIdxMultLoc->push_back(static_cast<unsigned int>(dofs[i]));
     }
     ierr = ISRestoreIndices(dofMultiplicities, &dofs); CHKERRQ(ierr);

     ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size); CHKERRQ(ierr);
     intersectLoc->reserve(size);
     for (int i = 0; i < size; i++) {
          auto *tmp = new vector<unsigned int>;
          ierr = ISGetLocalSize(dofIntersections[i], &m); CHKERRQ(ierr);
          ierr = ISGetIndices(dofIntersections[i], &dofs); CHKERRQ(ierr);
          for (int j = 0; j < m; j++) {
               tmp->push_back(static_cast<unsigned int>(dofs[j]));
          }
          ierr = ISRestoreIndices(dofIntersections[i], &dofs); CHKERRQ(ierr);
          intersectLoc->push_back(*tmp);
     }
     ierr = initGenEOPC(pc, N, n, rmap, P, NULL, NULL, localDofset, dofIdxMultLoc,
                        intersectLoc); CHKERRQ(ierr);
     PetscFunctionReturn(0);
}
}

/*
 * initGenEOPC: initialize the GenEO PC.
 *   - pcPC: PC created by createGenEOPC through PCRegister.
 *   - nbDOF, nbDOFLoc, pcMap, pcIS: local/global mapping (domain decomposition).
 *   - pcA: the A matrix (must be MatIS).
 *   - pcB: the right hand side.
 *   - pcX0: the initial guess.
 *           the user MUST ALWAYS call KSPSetInitialGuessNonzero(ksp, PETSC_TRUE) after initGenEOPC.
 *           pcX0 is set to a non zero initial guess if efficient hybrid mode is used.
 *           pcX0 is zeroed otherwise.
 *   - dofIdxDomLoc: set of DOF each local domain is made of.
 *   - dofIdxMultLoc: for each domain, list of DOF multiplicities.
 *                    the order of multiplicities must be consistent with the order of the DOF defined
 *                    by the local/global mapping.
 *   - intersectLoc: for each domain, list of intersections (DOF) with other domains.
 */
PetscErrorCode initGenEOPC(PC & pcPC,
                           unsigned int const & nbDOF, unsigned int const & nbDOFLoc,
                           ISLocalToGlobalMapping const & pcMap, Mat const & pcA, Vec const & pcB, Vec const & pcX0,
                           set<unsigned int> const * const dofIdxDomLoc, vector<unsigned int> const * const dofIdxMultLoc,
                           vector<vector<unsigned int>> const * const intersectLoc) {
  // Get the context.
  PetscErrorCode ierr;
  if (!pcPC) SETERRABT("GenEO preconditioner is invalid");
  geneoContext * gCtx = (geneoContext *) pcPC->data;
  if (!gCtx) SETERRABT("GenEO preconditioner without context");

  // Setup: inputs.

  gCtx->nbDOF = nbDOF;
  gCtx->nbDOFLoc = nbDOFLoc;
  gCtx->pcMap = pcMap;
  gCtx->pcA = pcA;
  gCtx->pcB = pcB;
  if (pcB) {
    ierr = PetscObjectReference((PetscObject)pcB); CHKERRQ(ierr);
  }
  gCtx->pcX0 = pcX0;
  if (pcX0) {
    ierr = PetscObjectReference((PetscObject)pcX0); CHKERRQ(ierr);
  }
  gCtx->pcIS = NULL;
  if (dofIdxDomLoc) {
    vector<PetscInt> pcIdxDomLoc;
    pcIdxDomLoc.reserve(nbDOFLoc);
    for (auto idx = dofIdxDomLoc->cbegin(); idx != dofIdxDomLoc->cend(); idx++) pcIdxDomLoc.push_back(*idx);
    PetscErrorCode pcRC = ISCreateGeneral(PETSC_COMM_WORLD, nbDOFLoc, pcIdxDomLoc.data(), PETSC_COPY_VALUES, &(gCtx->pcIS));
    CHKERRQ(pcRC);
  }
  gCtx->dofIdxMultLoc = dofIdxMultLoc; // Stored in the order imposed by the set.
  gCtx->intersectLoc = intersectLoc;

  return 0;
}

/*
 * createGenEOPC: create GenEO PC.
 * This function must be used as a callback passed to PCRegister.
 */
extern "C" {
PETSC_EXTERN PetscErrorCode createGenEOPC(PC pcPC) {
  // Create a context.

  geneoContext * gCtx = new geneoContext();
  if (!gCtx) SETERRABT("GenEO preconditioner without context");
  if (!pcPC) SETERRABT("GenEO preconditioner is invalid");
  pcPC->data = (void *) gCtx;

  // Default parameters.

  gCtx->lvl1ASM = true; // ASM.
  gCtx->lvl1RAS = gCtx->lvl1SRAS = gCtx->lvl1ORAS = false;
  gCtx->lvl2 = 1; // GenEO-1.
  gCtx->hybrid = false;
  gCtx->effHybrid = false;
  gCtx->optim = 0.;
  gCtx->tau   = 0.1; gCtx->tauLoc   = -1.;
  gCtx->gamma = 10.; gCtx->gammaLoc = -1.;
  gCtx->cst = false;
  gCtx->cut = -1;
  gCtx->noSyl = false;
  gCtx->offload = false;
  gCtx->debug = 0;     gCtx->debugBin = false; gCtx->debugMat = false;
  gCtx->check = false; gCtx->checkBin = false; gCtx->checkMat = false;

  // Setup: inputs.

  gCtx->nbDOF = 0;
  gCtx->nbDOFLoc = 0;
  gCtx->pcA = NULL;
  gCtx->pcMap = NULL;
  gCtx->pcIS = NULL;
  gCtx->dofIdxMultLoc = NULL;
  gCtx->intersectLoc = NULL;

  // Apply: Level 1.

  gCtx->pcXLoc = NULL;
  gCtx->pcScatCtx = NULL;
  gCtx->pcX = NULL;
  gCtx->pcXOld = NULL;
  gCtx->pcKSPL1Loc = NULL;
  gCtx->pcDLoc = NULL;

  // Apply: Level 2.

  gCtx->pcKSPL2 = NULL;
  gCtx->pcZE2G = NULL;
  gCtx->pcEEig = NULL;
  gCtx->pcYEig = NULL;
  gCtx->estimDimELoc = 0;
  gCtx->realDimELoc = 0;
  gCtx->nicolaidesLoc = 0;

  // Offload.

  gCtx->pcZE2GOff = NULL;
  gCtx->pcEEigOff = NULL;
  gCtx->pcKSPL2Off = NULL;
  gCtx->pcScatCtxOff = NULL;
  gCtx->pcXOff = NULL;
  gCtx->pcYEigOff = NULL;

  // Timers.

  gCtx->lvl1SetupMinvTimeLoc = 0.;
  gCtx->lvl2SetupTauLocTimeLoc = gCtx->lvl2SetupTauSylTimeLoc = gCtx->lvl2SetupTauEigTimeLoc = 0.;
  gCtx->lvl2SetupGammaLocTimeLoc = gCtx->lvl2SetupGammaSylTimeLoc = gCtx->lvl2SetupGammaEigTimeLoc = 0.;
  gCtx->lvl2SetupSylTimeLoc = gCtx->lvl2SetupEigTimeLoc = gCtx->lvl2SetupZTimeLoc = gCtx->lvl2SetupETimeLoc = 0.;
  gCtx->lvl1ApplyTimeLoc = 0.;
  gCtx->lvl1ApplyScatterTimeLoc = gCtx->lvl1ApplyMinvTimeLoc = gCtx->lvl1ApplyGatherTimeLoc = 0.;
  gCtx->lvl1ApplyPrjFSTimeLoc = gCtx->lvl1ApplyPrjFSZtTimeLoc = gCtx->lvl1ApplyPrjFSEinvTimeLoc = gCtx->lvl1ApplyPrjFSZTimeLoc = 0.;
  gCtx->lvl2ApplyTimeLoc = 0.;
  gCtx->lvl2ApplyZtTimeLoc = gCtx->lvl2ApplyEinvTimeLoc = gCtx->lvl2ApplyZTimeLoc = 0.;

  // Set callbacks.

  pcPC->ops->setup          = setUpGenEOPC;
  pcPC->ops->apply          = applyGenEOPC;
  pcPC->ops->destroy        = destroyGenEOPC;
  pcPC->ops->setfromoptions = setUpGenEOPCFromOptions;

  // Build GenEO name.

  PetscErrorCode pcRC = buildGenEOName(gCtx);
  CHKERRQ(pcRC);

  return 0;
}
}
