#ifndef __geneo__
#define __geneo__

#include <petsc.h>
#include <petsc/private/pcimpl.h> // Private include file intended for use by all preconditioner.

#include <set>
#include <vector>
#include <string>
#include <geneo_c.h>

using namespace std;

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
 *   - dofIdxDomLoc: set of (global) DOFs each local domain is made of.
 *   - dofIdxMultLoc: for each domain, list of DOF multiplicities.
 *                    the order of multiplicities must be consistent with the order of the DOF defined
 *                    by the local/global mapping.
 *   - intersectLoc: for each domain, list of intersections with other domains (list of local DOFs).
 */
PetscErrorCode initGenEOPC(PC & pcPC,
                           unsigned int const & nbDOF, unsigned int const & nbDOFLoc,
                           ISLocalToGlobalMapping const & pcMap, Mat const & pcA, Vec const & pcB, Vec const & pcX0,
                           vector<unsigned int> const * const dofIdxDomLoc, vector<unsigned int> const * const dofIdxMultLoc,
                           vector<vector<unsigned int>> const * const intersectLoc);

/*
 * usageGenEO: usage of GenEO.
 *   - petscPrintf: do or don't PetscPrintf (PETSC_COMM_WORLD).
 */
string usageGenEO(bool const petscPrintf = true);

/*
 * GenEO context.
 */
class geneoContext { // Shell preconditioner context: used to carry all informations necessary to build and use shell preconditioner.
  public:
    // GenEO name.

    string name;

    // GenEO parameters.

    bool lvl1ASM;
    bool lvl1RAS;
    bool lvl1SRAS;
    bool lvl1ORAS;
    int lvl2;
    bool hybrid;
    bool effHybrid;
    double optim; // Robin parameter for optimized methods.
    double tau; // GenEO parameter.
    double tauLoc; // GenEO local parameter.
    double gamma; // GenEO parameter.
    double gammaLoc; // GenEO local parameter.
    bool cst; // Do not allow local variations of GenEO parameters.
    int cut; // Cut-off the local number of eigen vectors.
    bool noSyl; // Do not use Sylvester's law of inertia to estimate the number of eigen values used to build Z.
    bool offload; // Offload Z and E at master side.

    // Setup: inputs.

    unsigned int nbDOF; // Size of the global domain.
    unsigned int nbDOFLoc; // Size of the local domain.
    ISLocalToGlobalMapping pcMap; // Index set: needed to create MatIS.
    Mat pcA; // A matrix (global matrix).
    Vec pcB; // Right hand side.
    Vec pcX0; // Initial guess.
    IS pcIS; // Index set: needed for scatter/gather (though this is the "same thing" than pcMap).
    vector<unsigned int> const * dofIdxMultLoc; // Multiplicity of DOFs of the local domain.
    vector<vector<unsigned int>> const * intersectLoc;

    // Apply: Level 1.

    Vec pcXLoc; // Local vector for scatter.
    VecScatter pcScatCtx; // Gather / scatter context.
    Vec pcX; // Global vector.
    Vec pcXOld; // Global vector.
    KSP pcKSPL1Loc; // Solve local dirichlet (LU) problem (level 1).
    Vec pcDLoc; // Local partition of unity.

    // Apply: level 2.

    KSP pcKSPL2; // Solve global coarse-projected (LU) problem (level 2): E^-1.
    Mat pcZE2G; // Coarse space (eigen space to global space): Z.
    Mat pcEEig; // Projection of A on the coarse space (eigen space to eigen space): E.
    Vec pcYEig;
    int estimDimELoc;
    int realDimELoc;
    int nicolaidesLoc;
    string infoL2; // Informations related to solves associated to the 2nd level (if any).

    // Offload.

    Mat * pcZE2GOff; // Local offload: Z.
    Mat * pcEEigOff; // Local offload: E.
    KSP pcKSPL2Off; // Local offload: E^-1.
    VecScatter pcScatCtxOff; // Local offload: gather / scatter context.
    Vec pcXOff;
    Vec pcYEigOff;

    // Timers.

    double lvl1SetupMinvTimeLoc;
    double lvl2SetupTauLocTimeLoc,   lvl2SetupTauSylTimeLoc,   lvl2SetupTauEigTimeLoc;
    double lvl2SetupGammaLocTimeLoc, lvl2SetupGammaSylTimeLoc, lvl2SetupGammaEigTimeLoc;
    double lvl2SetupSylTimeLoc,      lvl2SetupEigTimeLoc,      lvl2SetupZTimeLoc,        lvl2SetupETimeLoc;
    double lvl1ApplyTimeLoc;
    double lvl1ApplyScatterTimeLoc, lvl1ApplyMinvTimeLoc,    lvl1ApplyGatherTimeLoc;
    double lvl1ApplyPrjFSTimeLoc,   lvl1ApplyPrjFSZtTimeLoc, lvl1ApplyPrjFSEinvTimeLoc, lvl1ApplyPrjFSZTimeLoc;
    double lvl2ApplyTimeLoc;
    double lvl2ApplyZtTimeLoc, lvl2ApplyEinvTimeLoc, lvl2ApplyZTimeLoc;

    // Check.

    bool check;
    bool checkBin;
    bool checkMat;
    string checkFile;

    // Debug.

    int debug;
    bool debugBin;
    bool debugMat;
    string debugFile;
};

#endif
