/*
 * geneo4PETSc is an implementation of the GenEO preconditioner with PETSc and SLEPc.
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
 *   - all variables suffixed "Loc" are associated to local matrices (domain).
 *     Note: conversely, all variables that are not suffixed "Loc" are associated to the global problem.
 *
 * notes:
 *   - A can be made from index sets (IS) that may overlap, it is built using MatIS.
 *   - when building a MatIS:
 *     - first, a local matrix is created (preallocated, filled) per domain (MPI proc).
 *     - then, local matrices are gathered by the (global/distributed) MatIS (MatISSetLocalMat).
 *   - some operations are NOT supported on MatIS (in particular PtAP): if needed, a MatMPI version
 *     (which supports PtAP) of a MatIS matrix may be created (MatISGetMPIXAIJ).
 */

#include <iostream> // cerr.
#include <string>
#include <fstream> // ifstream, ofstream.
#include <sstream> // stringstream.
#include <vector>
#include <set>
#include <iomanip> // setprecision.
#include <chrono>
#include <algorithm> // set_intersection, max_element, min_element.
#include <iterator> // inserter.

#include <dlfcn.h> // dlopen, dlsym, dlclose.

#include <petsc.h>
#include <slepc.h>

#include "geneo.hpp"

#include "metis.h"

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>

using namespace std;

class options {
  public:
    string inpFileA;
    string inpFileB;
    double inpEps;
    string inpLibA;
    string inpLibArg;
    bool metisDual;
    bool useMatMPI;
    unsigned int addOverlap;
    bool debug;
    bool debugBin;
    bool debugMat;
    int verbose;
    bool timing;
    bool shortRes;
    bool cmdLine;
    stringstream userCmdLine;
};

int getLibInput(string const & inpLibA, string const & inpLibArg,
                unsigned int & nbElem, unsigned int & nbNode, vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx,
                vector<vector<PetscScalar>> & elemSubMat) {
  void * lib = dlopen(inpLibA.c_str(), RTLD_LAZY);
  if (!lib) {cerr << "Error: open library KO - " << dlerror() << endl; return 1;}

  typedef int (*libFunction) (std::string const & args,
                              unsigned int & nbElem, unsigned int & nbNode,
                              std::vector<unsigned int> & elemPtr, std::vector<unsigned int> & elemIdx,
                              std::vector<std::vector<double>> & elemSubMat);
  libFunction getInp = (libFunction) dlsym(lib, "getInput");
  if (!getInp) {cerr << "Error: get input function from library KO - " << dlerror() << endl; dlclose(lib); return 1;}

  string args = inpLibArg;
  while (args.find("#") != string::npos) {args.replace(args.find("#"), 1, " ");}
  int rc = (*getInp)(args, nbElem, nbNode, elemPtr, elemIdx, elemSubMat);
  if (rc != 0) {cerr << "Error: get input data from library KO" << endl; dlclose(lib); return 1;}

  dlclose(lib);

  return 0;
}

int readLineFile(string const & inpLine,
                 vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx, set<unsigned int> & nSet,
                 double const & inpEps, vector<vector<PetscScalar>> & elemSubMat) {
  // Get element in CSR format (pointer, index).

  bool fillDOF = true;
  vector<unsigned int> elemDOF;
  vector<PetscScalar> elemMat;
  stringstream inpSS(inpLine);
  elemPtr.push_back(elemIdx.size());
  while (inpSS) {
    string token; inpSS >> token;
    if (!inpSS) break;
    if (token == "-") {fillDOF = false; continue;} // Now, fill matrix.

    stringstream tokenSS(token);
    if (fillDOF) {
      unsigned int d = 0; tokenSS >> d;
      if (tokenSS) {
        elemDOF.push_back(d);
        elemIdx.push_back(d); // Always inserted.
        nSet.insert(d); // Not inserted if already present.
      }
    }
    else { // Provide a user-defined matrix.
      double aij = 0.; tokenSS >> aij;
      if (tokenSS) elemMat.push_back(aij);
    }
  }

  // Provide default matrix if necessary.

  if (elemMat.size() == 0) { // Provide a default matrix.
    unsigned int const nDOF = elemDOF.size();
    for (unsigned int i = 0; i < nDOF; i++) {
      for (unsigned int j = 0; j < nDOF; j++) {
        if (i == j) elemMat.push_back( 1.+inpEps);
        else        elemMat.push_back(-1./((double) (nDOF-1)));
      }
    }
  }
  elemSubMat.push_back(elemMat);

  return 0;
}

int readInputFile(string const & inpFileA,
                  unsigned int & nbElem, unsigned int & nbNode, vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx,
                  double const & inpEps, vector<vector<PetscScalar>> & elemSubMat) {
  // Read input file.

  nbElem = nbNode = 0;
  ifstream inp(inpFileA);
  if (!inp) {cerr << "Error: can not open " << inpFileA << endl; return 1;}
  set<unsigned int> nSet; // Sorted set of unique nodes.
  do {
    // Skip comments.

    string inpLine; getline(inp, inpLine);
    while (isspace(*inpLine.begin())) inpLine.erase(inpLine.begin()); // Suppress leading white spaces.
    if (inpLine.length() == 0) continue; // Empty line.
    if (inpLine[0] == '%' || inpLine[0] == '#') continue; // Comments skipped, begin reading.

    // Read line by line.

    int rc = readLineFile(inpLine, elemPtr, elemIdx, nSet, inpEps, elemSubMat);
    if (rc != 0) {cerr << "Error: read line file KO" << endl; return 1;}
    nbElem++;
  }
  while (inp);
  elemPtr.push_back(elemIdx.size());
  nbNode = nSet.size();

  // Check validity.

  unsigned int idxMax = *max_element(nSet.cbegin(), nSet.cend()); // If the set contains n nodes, 0...n-1 must be contained in the set.
  if (idxMax+1 != nbNode) {
    for (unsigned int idx = 0; idx < nSet.size(); idx++) {
      if (nSet.find(idx) == nSet.end()) cerr << "Error: " << idx << " not found in node set." << endl;
    }
    cerr << "Error: bad node set (" << idxMax+1 << "/" << nbNode << ") in file " << endl; return 1;
  }

  unsigned int const nElem = elemSubMat.size();
  for (unsigned int e = 0; e < nbElem; e++) {
    unsigned int const startElemIdx = elemPtr[e];
    unsigned int const nbNodePerElem = elemPtr[e+1]-startElemIdx;
    for (unsigned int n = 0; n < nbNodePerElem; n++) {
      unsigned int nIdx = elemIdx[startElemIdx+n];
      if (nIdx >= nbNode) {cerr << "Error: bad element (" << e+1 << ", " << nIdx << "/" << nbNode << ") in file " << endl; return 1;}
    }
    unsigned int const subMatSize = nbNodePerElem*nbNodePerElem;
    if (e >= nElem || elemSubMat[e].size() != subMatSize) {cerr << "Error: bad matrix (" << e+1 << ") in file " << endl; return 1;}
  }

  return 0;
}

int buildElemPartFromNodePart(bool const & metisDual, unsigned int const & nbPart, vector<unsigned int> const & nodePart,
                              vector<unsigned int> const & elemPtr, vector<unsigned int> const & elemIdx,
                              unsigned int const & p, vector<unsigned int> & ePart) {
  if (metisDual) return 0; // Element partition is OK: it is the one computed by metis.

  // If metis nodal mode, build element partition.

  for (unsigned int e = 0; e < ePart.size(); e++) {
    ePart[e] = nbPart; // Out of all partitions.

    unsigned int const startElemIdx = elemPtr[e];
    unsigned int const nbNodePerElem = elemPtr[e+1]-startElemIdx;
    for (unsigned int n = 0; n < nbNodePerElem; n++) {
      unsigned int nIdx = elemIdx[startElemIdx+n];
      if (nodePart[nIdx] == p) ePart[e] = p; // Element belongs to the partition if one of its nodes does.
    }
  }

  return 0;
}

int computeInverseTopology(vector<set<unsigned int>> & invTopo,
                           unsigned int const & nbNode, unsigned int const & nbElem,
                           vector<unsigned int> const & elemPtr, vector<unsigned int> const & elemIdx) {
  // Compute inverse topology.

  invTopo.assign(nbNode, set<unsigned int>());
  for (unsigned int e = 0; e < nbElem; e++) {
    unsigned int const startElemIdx = elemPtr[e];
    unsigned int const nbNodePerElem = elemPtr[e+1]-startElemIdx;
    for (unsigned int n = 0; n < nbNodePerElem; n++) {
      unsigned int nIdx = elemIdx[startElemIdx+n];
      if (nIdx >= nbNode) {cerr << "Error: bad node, compute inverse topology KO" << endl; return 1;}
      if (invTopo[nIdx].find(e) == invTopo[nIdx].end()) { // Element not already added in the set.
        invTopo[nIdx].insert(e); // Match: node <-> element.
      }
    }
  }

  return 0;
}

int addOverlapLayers(options const & opt, unsigned int const & p, vector<set<unsigned int>> const & invTopo,
                     unsigned int const & nbElem, vector<unsigned int> const & elemPtr, vector<unsigned int> const & elemIdx,
                     vector<unsigned int> & ePart) {
  // Add overlap layers.

  unsigned int addOverlap = opt.addOverlap;
  while(addOverlap > 0) {
    set<unsigned int> newElemSet;

    for (unsigned int e = 0; e < ePart.size(); e++) {
      if (ePart[e] != p) continue;

      unsigned int const startElemIdx = elemPtr[e];
      unsigned int const nbNodePerElem = elemPtr[e+1]-startElemIdx;
      for (unsigned int n = 0; n < nbNodePerElem; n++) {
        unsigned int nIdx = elemIdx[startElemIdx+n];
        for (auto i = invTopo[nIdx].cbegin(); i != invTopo[nIdx].cend(); i++) {
          unsigned int const eIdx = *i; // Index of the element.
          if (eIdx >= nbElem) {cerr << "Error: bad element, add overlap layer KO" << endl; return 1;}
          if (ePart[eIdx] != p && newElemSet.find(eIdx) == newElemSet.end()) newElemSet.insert(eIdx);
        }
      }
    }

    for (auto i = newElemSet.cbegin(); i != newElemSet.cend(); i++) {
      unsigned int const e = *i; // Index of the element.
      if (e >= nbElem) {cerr << "Error: bad element, add overlap layer KO" << endl; return 1;}
      ePart[e] = p; // Add element.
    }

    addOverlap--;
  }

  // Debug on demand.

  if (opt.debug) {
    ofstream debug;
    if (p == 0) debug.open("debug.input.overlap.log", ios::out);
    else        debug.open("debug.input.overlap.log", ios::app);
    for (unsigned int e = 0; e < ePart.size(); e++) {
      if (ePart[e] != p) continue;

      debug << "elem: ";
      unsigned int const startElemIdx = elemPtr[e];
      unsigned int const nbNodePerElem = elemPtr[e+1]-startElemIdx;
      for (unsigned int n = 0; n < nbNodePerElem; n++) debug << elemIdx[startElemIdx+n] << " ";
      debug << "=> partition: " << ePart[e] << endl;
    }
    debug << endl;
  }

  return 0;
}

int decompose(unsigned int const & nbPart, unsigned int const & nbNode, unsigned int const & nbElem,
              vector<set<unsigned int>> & nodeIdxDom, vector<unsigned int> & nodeIdxMult,
              vector<set<unsigned int>> & elemIdxDom, vector<unsigned int> & elemIdxMult,
              vector<vector<vector<unsigned int>>> & intersectDom,
              options const & opt, vector<unsigned int> const & elemPart, vector<unsigned int> const & nodePart,
              vector<unsigned int> const & elemPtr, vector<unsigned int> const & elemIdx) {
  // Compute inverse topology on demand.

  vector<set<unsigned int>> invTopo;
  if (opt.addOverlap) {
    int rc = computeInverseTopology(invTopo, nbNode, nbElem, elemPtr, elemIdx);
    if (rc != 0) {cerr << "Error: compute inverse topology KO" << endl; return 1;}
  }

  // Build domains from partitions.

  nodeIdxDom.assign(nbPart, set<unsigned int>());
  nodeIdxMult.assign(nbNode, 0);
  elemIdxDom.assign(nbPart, set<unsigned int>());
  elemIdxMult.assign(nbElem, 0);
  for (unsigned int p = 0; p < nbPart; p++) {
    // Build element partition.

    vector<unsigned int> ePart = elemPart;
    int rc = buildElemPartFromNodePart(opt.metisDual, nbPart, nodePart, elemPtr, elemIdx, p, ePart);
    if (rc != 0) {cerr << "Error: build element partition KO" << endl; return 1;}

    // Add overlap on demand.

    rc = addOverlapLayers(opt, p, invTopo, nbElem, elemPtr, elemIdx, ePart);
    if (rc != 0) {cerr << "Error: add overlap KO" << endl; return 1;}

    // Build domain from partition (domain = partition + borders).

    for (unsigned int e = 0; e < ePart.size(); e++) {
      if (ePart[e] != p) continue;

      if (elemIdxDom[p].find(e) == elemIdxDom[p].end()) { // Element not already added in the domain.
        elemIdxDom[p].insert(e); // Element stored in the domain.
        elemIdxMult[e] += 1; // Compute weight of elements shared by several partitions.
      }

      unsigned int const startElemIdx = elemPtr[e];
      unsigned int const nbNodePerElem = elemPtr[e+1]-startElemIdx;
      for (unsigned int n = 0; n < nbNodePerElem; n++) {
        unsigned int nIdx = elemIdx[startElemIdx+n];
        if (nIdx >= nbNode) {cerr << "Error: build domain from partition KO" << endl; return 1;}
        if (nodeIdxDom[p].find(nIdx) == nodeIdxDom[p].end()) { // Node not already added in the domain.
          nodeIdxDom[p].insert(nIdx); // Node stored in the domain.
          nodeIdxMult[nIdx] += 1; // Compute partition of unity: (1.25) from R1.
        }
      }
    }
  }

  // Build domain intersections.

  intersectDom.assign(nbPart, vector<vector<unsigned int>>());
  for (unsigned int p = 0; p < nbPart; p++) {
    intersectDom[p].resize(nbPart); // p will have at most nbPart intersections (one with each potential neighbor).
    for (unsigned int q = 0; q < nbPart; q++) intersectDom[p][q].clear();

    for (unsigned int q = 0; q < nbPart; q++) {
      if (p == q) continue; // Skip self intersection !

      // Intersection as a set of global indices.

      set<unsigned int> intersectPQ;
      set_intersection(nodeIdxDom[p].begin(), nodeIdxDom[p].end(), nodeIdxDom[q].begin(), nodeIdxDom[q].end(), // p inter q.
                       inserter(intersectPQ, intersectPQ.begin()));
      if (intersectPQ.empty()) continue;

      // Intersection as a set of local indices.

      vector<unsigned int> intersectPQLoc;
      intersectPQLoc.reserve(intersectPQ.size());
      for (auto idx = intersectPQ.cbegin(); idx != intersectPQ.cend(); idx++) { // Stored in the order imposed by the set.
        auto found = nodeIdxDom[p].find(*idx);
        if (found == nodeIdxDom[p].end()) {cerr << "Error: global index not found in local domain" << endl; return 1;}
        unsigned int idxLoc = distance(nodeIdxDom[p].begin(), found); // Local index in the set.
        intersectPQLoc.push_back(idxLoc); // Use local index: point directly to index mult. or unity part. in domains.
      }
      intersectDom[p][q] = intersectPQLoc;
    }
  }

  return 0;
}

int partition(bool const & metisDual,
              unsigned int & nbElem, unsigned int & nbNode, vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx, unsigned int nbPart,
              vector<unsigned int> & elemPart, vector<unsigned int> & nodePart, int const & debug) {
  // Partition with metis.

  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_MINCONN] = 1;
  options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;

  idx_t obj = 0;
  idx_t * pElemPart = new idx_t[nbElem];
  idx_t * pNodePart = new idx_t[nbNode];
  idx_t nCommon = 1;
  auto start = chrono::high_resolution_clock::now();
  if (nbPart == 1) {
    for (unsigned int n = 0; n < nbNode; n++) pNodePart[n] = 0; // Do not call metis (crash if 1 partition).
    for (unsigned int e = 0; e < nbElem; e++) pElemPart[e] = 0; // Do not call metis (crash if 1 partition).
  }
  else {
    idx_t nElem = nbElem, nNode = nbNode, nPart = nbPart;
    vector<idx_t> ePtr(elemPtr.size()); for (unsigned int e = 0; e < elemPtr.size(); e++) ePtr[e] = elemPtr[e];
    vector<idx_t> eIdx(elemIdx.size()); for (unsigned int e = 0; e < elemIdx.size(); e++) eIdx[e] = elemIdx[e];
    if (metisDual) {
      int rc = METIS_PartMeshDual(&nElem, &nNode, ePtr.data(), eIdx.data(), NULL, NULL, &nCommon, &nPart,
                                  NULL, options, &obj, pElemPart, pNodePart);
      if (rc != METIS_OK) {cerr << "Error: METIS_PartMeshDual KO" << endl; return 1;}
    }
    else {
      int rc = METIS_PartMeshNodal(&nElem, &nNode, ePtr.data(), eIdx.data(), NULL, NULL, &nPart,
                                   NULL, options, &obj, pElemPart, pNodePart);
      if (rc != METIS_OK) {cerr << "Error: METIS_PartMeshNodal KO" << endl; return 1;}
    }
  }
  auto stop = chrono::high_resolution_clock::now();

  nodePart.assign(nbNode, -1); for (unsigned int n = 0; n < nbNode; n++) nodePart[n] = pNodePart[n];
  elemPart.assign(nbElem, -1); for (unsigned int e = 0; e < nbElem; e++) elemPart[e] = pElemPart[e];
  if (pElemPart) {delete [] pElemPart; pElemPart = NULL;}
  if (pNodePart) {delete [] pNodePart; pNodePart = NULL;}

  // Debug on demand.

  if (debug >= 2) {
    auto time = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
    ofstream debug("debug.input.metis.log");
    if (metisDual) {
      debug << "Element partition (computed in " << setprecision(7) << time << " s) is :" << endl << endl;
      for (unsigned int e = 0; e < nbElem; e++) {
        debug << "elem: ";
        unsigned int const startElemIdx = elemPtr[e];
        unsigned int const nbNodePerElem = elemPtr[e+1]-startElemIdx;
        for (unsigned int n = 0; n < nbNodePerElem; n++) debug << elemIdx[startElemIdx+n] << " ";
        debug << "=> partition: " << elemPart[e] << endl;
      }
    }
    else {
      debug << "Node partition (computed in " << setprecision(7) << time << " s) is :" << endl << endl;
      for (unsigned int n = 0; n < nbNode; n++) debug << "node: " << n << " => partition: " << nodePart[n] << endl;
    }
  }

  return 0;
}

int buildDomain(unsigned int const & nbPart, unsigned int const & nbNode, unsigned int const & nbElem, unsigned int const & p,
                vector<unsigned int> const & elemPtr, vector<unsigned int> const & elemIdx, vector<vector<PetscScalar>> const & elemSubMat,
                vector<set<unsigned int>> const & nodeIdxDom, vector<unsigned int> const & nodeIdxMult,
                vector<set<unsigned int>> const & elemIdxDom, vector<unsigned int> const & elemIdxMult,
                unsigned int & nbNodeDom, unsigned int & nbElemDom,
                vector<unsigned int> & ePtrDom, vector<unsigned int> & eIdxDom,
                vector<vector<PetscScalar>> & eSubMatDom, vector<unsigned int> & nIdxMultDom) {
  // Build domain.

  ePtrDom.reserve(elemPtr.size()/nbPart);
  eIdxDom.reserve(elemIdx.size()/nbPart);
  eSubMatDom.reserve(nbElem/nbPart);
  for (auto i = elemIdxDom[p].cbegin(); i != elemIdxDom[p].cend(); i++) {
    unsigned int const e = *i; // Index of the element.
    if (e >= nbElem) {cerr << "Error: bad element, build domain KO" << endl; return 1;}

    // Build domain elements.

    ePtrDom.push_back(eIdxDom.size());

    unsigned int const startElemIdx = elemPtr[e];
    unsigned int const nbNodePerElem = elemPtr[e+1]-startElemIdx;
    for (unsigned int n = 0; n < nbNodePerElem; n++) eIdxDom.push_back(elemIdx[startElemIdx+n]);

    // Element partition of unity: weight elements shared by several partitions.

    PetscScalar elemWeightPart = 1./((double) elemIdxMult[e]);
    vector<PetscScalar> elemSubMatWgt = elemSubMat[e];
    for (unsigned int m = 0; m < elemSubMatWgt.size(); m++) elemSubMatWgt[m] *= elemWeightPart;
    eSubMatDom.push_back(elemSubMatWgt);

    nbElemDom++;
  }
  ePtrDom.push_back(eIdxDom.size());

  // Node multiplicity.

  nIdxMultDom.reserve(nodeIdxDom[p].size());
  for (auto i = nodeIdxDom[p].cbegin(); i != nodeIdxDom[p].cend(); i++) {
    unsigned int const n = *i; // Index of the node.
    if (n >= nbNode) {cerr << "Error: bad node, build domain KO" << endl; return 1;}
    nIdxMultDom.push_back(nodeIdxMult[n]); // Stored in the order imposed by the set.
  }

  nbNodeDom = nodeIdxDom[p].size();

  return 0;
}

int sendDomain(unsigned int const & nbPart, unsigned int const & nbNode, unsigned int const & nbElem,
               vector<unsigned int> const & elemPtr, vector<unsigned int> const & elemIdx, vector<vector<PetscScalar>> const & elemSubMat,
               vector<set<unsigned int>> const & nodeIdxDom, vector<unsigned int> const & nodeIdxMult,
               vector<set<unsigned int>> const & elemIdxDom, vector<unsigned int> const & elemIdxMult,
               unsigned int & nbNodeLoc, unsigned int & nbElemLoc,
               vector<unsigned int> & elemPtrLoc, vector<unsigned int> & elemIdxLoc, vector<vector<PetscScalar>> & elemSubMatLoc,
               set<unsigned int> & nodeIdxDomLoc, vector<unsigned int> & nodeIdxMultLoc,
               vector<vector<vector<unsigned int>>> const & intersectDom, vector<vector<unsigned int>> & intersectLoc) {
  // Send domains.

  bool wrng = false;
  for (unsigned int p = 0; p < nbPart; p++) {
    // Build domain.

    unsigned int nbNodeDom = 0, nbElemDom = 0;
    vector<unsigned int> ePtrDom, eIdxDom;
    vector<vector<PetscScalar>> eSubMatDom;
    vector<unsigned int> nIdxMultDom;
    int rc = buildDomain(nbPart, nbNode, nbElem, p,
                         elemPtr, elemIdx, elemSubMat, nodeIdxDom, nodeIdxMult, elemIdxDom, elemIdxMult,
                         nbNodeDom, nbElemDom, ePtrDom, eIdxDom, eSubMatDom, nIdxMultDom);
    if (rc != 0) {cerr << "Error: build domain KO" << endl; return 1;}
    if (nbNodeDom == 0) {cout << "WRNG: the domain " << p << " is empty" << endl; wrng = true;}

    // Send inputs to each process.

    if (p == 0) {
      nbNodeLoc = nbNodeDom;
      nbElemLoc = nbElemDom;
      elemPtrLoc = ePtrDom;
      elemIdxLoc = eIdxDom;
      elemSubMatLoc = eSubMatDom;
      nodeIdxDomLoc = nodeIdxDom[p];
      nodeIdxMultLoc = nIdxMultDom;
      intersectLoc = intersectDom[p];
    }
    else {
      boost::mpi::communicator world;
      world.send(p, 0, nbNode);
      world.send(p, 1, nbElem);
      world.send(p, 2, nbNodeDom);
      world.send(p, 3, nbElemDom);
      world.send(p, 4, ePtrDom);
      world.send(p, 5, eIdxDom);
      world.send(p, 6, eSubMatDom);
      world.send(p, 7, nodeIdxDom[p]);
      world.send(p, 8, nIdxMultDom);
      world.send(p, 9, intersectDom[p]);
    }
  }
  if (wrng) cout << endl; // Separate WRNG from INFO in output.

  return 0;
}

int recvDomain(unsigned int & nbNode, unsigned int & nbElem, unsigned int & nbNodeLoc, unsigned int & nbElemLoc,
               vector<unsigned int> & elemPtrLoc, vector<unsigned int> & elemIdxLoc, vector<vector<PetscScalar>> & elemSubMatLoc,
               set<unsigned int> & nodeIdxDomLoc, vector<unsigned int> & nodeIdxMultLoc, vector<vector<unsigned int>> & intersectLoc) {
  // Receive partitions from master.

  boost::mpi::communicator world;
  world.recv(0, 0, nbNode);
  world.recv(0, 1, nbElem);
  world.recv(0, 2, nbNodeLoc);
  world.recv(0, 3, nbElemLoc);
  world.recv(0, 4, elemPtrLoc);
  world.recv(0, 5, elemIdxLoc);
  world.recv(0, 6, elemSubMatLoc);
  world.recv(0, 7, nodeIdxDomLoc);
  world.recv(0, 8, nodeIdxMultLoc);
  world.recv(0, 9, intersectLoc);

  return 0;
}

int partitionAndDecompose(unsigned int & nbNode, unsigned int & nbElem, unsigned int & nbNodeLoc, unsigned int & nbElemLoc,
                          vector<unsigned int> & elemPtrLoc, vector<unsigned int> & elemIdxLoc, vector<vector<PetscScalar>> & elemSubMatLoc,
                          set<unsigned int> & nodeIdxDomLoc, vector<unsigned int> & nodeIdxMultLoc,
                          vector<vector<unsigned int>> & intersectLoc, double & readInpTime, double & partDecompTime, options const & opt) {
  // Partition (done by the master, then, sent by master to slaves).

  boost::mpi::communicator world;
  int const rank = world.rank();
  if (rank == 0) {
    // Read input file.

    auto start = chrono::high_resolution_clock::now();
    vector<unsigned int> elemPtr, elemIdx;
    vector<vector<PetscScalar>> elemSubMat;
    if (opt.inpFileA.length() > 0) {
      int rc = readInputFile(opt.inpFileA, nbElem, nbNode, elemPtr, elemIdx, opt.inpEps, elemSubMat);
      if (rc != 0) {cerr << "Error: read input file KO" << endl; return 1;}
    }
    else {
      int rc = getLibInput(opt.inpLibA, opt.inpLibArg, nbElem, nbNode, elemPtr, elemIdx, elemSubMat);
      if (rc != 0) {cerr << "Error: get input from library KO" << endl; return 1;}
    }
    if (nbNode == 0 || nbElem == 0) {cerr << "Error: empty input" << endl; return 1;}
    if (nbElem+1 != elemPtr.size()) {cerr << "Error: bad input pointer" << endl; return 1;}
    if (elemIdx.size() != elemPtr[nbElem]) {cerr << "Error: bad input pointer / index" << endl; return 1;}
    if (nbElem != elemSubMat.size()) {cerr << "Error: bad input sub matrices" << endl; return 1;}
    auto stop = chrono::high_resolution_clock::now();
    readInpTime = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

    // Partition and decompose.

    start = chrono::high_resolution_clock::now();
    vector<unsigned int> elemPart, nodePart;
    unsigned int const nbPart = world.size();
    int rc = partition(opt.metisDual, nbElem, nbNode, elemPtr, elemIdx, nbPart, elemPart, nodePart, opt.debug);
    if (rc != 0) {cerr << "Error: partition KO" << endl; return 1;}

    vector<set<unsigned int>> nodeIdxDom; // Global indices of each domain node.
    vector<set<unsigned int>> elemIdxDom; // Global indices of each domain element.
    vector<unsigned int> nodeIdxMult; // Number of domain(s) each node belongs to (multiplicity per node).
    vector<unsigned int> elemIdxMult; // Number of domain(s) each element belongs to (multiplicity per element).
    vector<vector<vector<unsigned int>>> intersectDom; // Intersections between each couple of domains.
    rc = decompose(nbPart, nbNode, nbElem, nodeIdxDom, nodeIdxMult, elemIdxDom, elemIdxMult, intersectDom,
                   opt, elemPart, nodePart, elemPtr, elemIdx);
    if (rc != 0) {cerr << "Error: decompose KO" << endl; return 1;}
    stop = chrono::high_resolution_clock::now();
    partDecompTime = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

    // Send inputs to all slaves.

    rc = sendDomain(nbPart, nbNode, nbElem, elemPtr, elemIdx, elemSubMat,
                    nodeIdxDom, nodeIdxMult, elemIdxDom, elemIdxMult,
                    nbNodeLoc, nbElemLoc, elemPtrLoc, elemIdxLoc, elemSubMatLoc, nodeIdxDomLoc, nodeIdxMultLoc,
                    intersectDom, intersectLoc);
    if (rc != 0) {cerr << "Error: send domain KO" << endl; return 1;}
  }
  else {
    int rc = recvDomain(nbNode, nbElem,
                        nbNodeLoc, nbElemLoc, elemPtrLoc, elemIdxLoc, elemSubMatLoc, nodeIdxDomLoc, nodeIdxMultLoc,
                        intersectLoc);
    if (rc != 0) {cerr << "Error: recv domain KO" << endl; return 1;}
  }

  if (nodeIdxDomLoc.size() != nbNodeLoc) {cerr << "Error[" << rank << "]: bad domain index" << endl; return 1;}
  if (nodeIdxMultLoc.size() != nbNodeLoc) {cerr << "Error[" << rank << "]: bad domain multiplicity" << endl; return 1;}
  if (nbElemLoc+1 != elemPtrLoc.size()) {cerr << "Error[" << rank << "]: bad domain pointer" << endl; return 1;}
  if (elemIdxLoc.size() != elemPtrLoc[nbElemLoc]) {cerr << "Error[" << rank << "]: bad domain pointer / index" << endl; return 1;}
  if (nbElemLoc != elemSubMatLoc.size()) {cerr << "Error[" << rank << "]: bad domain sub matrices" << endl; return 1;}

  return 0;
}

int preallocateALoc(unsigned int const & nbDOFLoc, unsigned int const & nbSubMatLoc,
                    vector<unsigned int> const & subMatPtrLoc, vector<unsigned int> const & subMatIdxLoc,
                    set<unsigned int> const & dofIdxDomLoc, Mat & pcALoc, unsigned int & nbNonNullValLoc) {
  nbNonNullValLoc = 0;

  // Preallocate the local part of A.

  vector<set<PetscInt>> ijInDom; // Track already inserted values at (i, j) to determine preallocation.
  ijInDom.assign(nbDOFLoc, set<PetscInt>());
  for (unsigned int e = 0; e < nbSubMatLoc; e++) {
    vector<PetscInt> pcIdx;
    unsigned int const startSubMatIdxLoc = subMatPtrLoc[e];
    unsigned int const nbDOFPerSubMat = subMatPtrLoc[e+1]-startSubMatIdxLoc;
    for (unsigned int n = 0; n < nbDOFPerSubMat; n++) {
      unsigned int idx = subMatIdxLoc[startSubMatIdxLoc+n]; // Global index of each dimension of each value.
      auto found = dofIdxDomLoc.find(idx);
      if (found == dofIdxDomLoc.end()) {cerr << "Error: global index not found in local domain" << endl; return 1;}
      PetscInt pcIdxLoc = distance(dofIdxDomLoc.begin(), found); // Local index of each dimension.
      pcIdx.push_back(pcIdxLoc); // Use local index as domain matrix are set locally.
    }

    for (auto pcIdxI = pcIdx.cbegin(); pcIdxI != pcIdx.cend(); pcIdxI++) {
      for (auto pcIdxJ = pcIdx.cbegin(); pcIdxJ != pcIdx.cend(); pcIdxJ++) {
        ijInDom[*pcIdxI].insert(*pcIdxJ); // Need to forecast space for all (i, j) values.
      }
    }
  }
  PetscInt * pcNbNonNullValPerRowLoc = new PetscInt[nbDOFLoc]; // Number of non null values per row of the local part of A.
  for (unsigned int i = 0; i < nbDOFLoc; i++) {
    pcNbNonNullValPerRowLoc[i] = ijInDom[i].size();
    nbNonNullValLoc += ijInDom[i].size();
  }

  PetscErrorCode pcRC = MatCreateSeqAIJ(PETSC_COMM_SELF, nbDOFLoc, nbDOFLoc, 0, pcNbNonNullValPerRowLoc, &pcALoc);
  CHKERRQ(pcRC);
  if (pcNbNonNullValPerRowLoc) {delete [] pcNbNonNullValPerRowLoc; pcNbNonNullValPerRowLoc = NULL;}

  return 0;
}

int fillALoc(unsigned int const & nbSubMatLoc,
             vector<unsigned int> const & subMatPtrLoc, vector<unsigned int> const & subMatIdxLoc,
             vector<vector<PetscScalar>> const & subMatLoc, set<unsigned int> const & dofIdxDomLoc, Mat & pcALoc) {
  // Fill the local part of A.

  for (unsigned int e = 0; e < nbSubMatLoc; e++) {
    // Get the indices of the local part of A.

    vector<PetscInt> pcIdx;
    unsigned int const startSubMatIdxLoc = subMatPtrLoc[e];
    unsigned int const nbDOFPerSubMat = subMatPtrLoc[e+1]-startSubMatIdxLoc;
    for (unsigned int n = 0; n < nbDOFPerSubMat; n++) {
      unsigned int idx = subMatIdxLoc[startSubMatIdxLoc+n]; // Global index of each dimension of each value.
      auto found = dofIdxDomLoc.find(idx);
      if (found == dofIdxDomLoc.end()) {cerr << "Error: global index not found in local domain" << endl; return 1;}
      PetscInt pcIdxLoc = distance(dofIdxDomLoc.begin(), found); // Local index of each dimension.
      pcIdx.push_back(pcIdxLoc); // Use local index as domain matrix are set locally.
    }

    // Fill A with an elementary matrix for each element in the partition.

    PetscInt pcNbIdx = pcIdx.size();
    PetscErrorCode pcRC = MatSetValues(pcALoc, pcNbIdx, pcIdx.data(), pcNbIdx, pcIdx.data(),
                                       subMatLoc[e].data(), ADD_VALUES);
    CHKERRQ(pcRC);
  }
  PetscErrorCode pcRC = MatAssemblyBegin(pcALoc, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);
  pcRC = MatAssemblyEnd(pcALoc, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);

  return 0;
}

PetscErrorCode createPetscViewer(bool const & bin, bool const & mat, MPI_Comm const & comm,
                                 string const & baseName, PetscViewer & pcView) {
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

int createA(unsigned int const & nbDOFLoc, unsigned int const & nbSubMatLoc,
            vector<unsigned int> const & subMatPtrLoc, vector<unsigned int> const & subMatIdxLoc,
            vector<vector<PetscScalar>> const & subMatLoc, set<unsigned int> const & dofIdxDomLoc,
            unsigned int const & nbDOF, ISLocalToGlobalMapping const & pcMap, Mat & pcA,
            unsigned int & nbNonNullValLoc, options const & opt) {
  // Create the local part of A.

  Mat pcALoc; // Local part of A.
  int rc = preallocateALoc(nbDOFLoc, nbSubMatLoc, subMatPtrLoc, subMatIdxLoc, dofIdxDomLoc, pcALoc, nbNonNullValLoc);
  if (rc != 0) {cerr << "Error: preallocate A KO" << endl; return 1;}
  rc = fillALoc(nbSubMatLoc, subMatPtrLoc, subMatIdxLoc, subMatLoc, dofIdxDomLoc, pcALoc);
  if (rc != 0) {cerr << "Error: fill A KO" << endl; return 1;}

  // Create A (globally by aggregation of its local parts).

  // MatIS means that the matrix is not assembled. The easiest way to think of this is that processes do not have to hold
  // full rows. One process can hold part of row i, and another processes can hold another part. The local size here is
  // not the size of the local IS block, since that is a property only of MatIS. It is the size of the local piece of the
  // vector you multiply. This allows PETSc to understand the parallel layout of the Vec, and how it matched the Mat.
  PetscErrorCode pcRC = MatCreateIS(PETSC_COMM_WORLD, 1, PETSC_DECIDE, PETSC_DECIDE, nbDOF, nbDOF, pcMap, pcMap, &pcA);
  CHKERRQ(pcRC);
  pcRC = MatISSetLocalMat(pcA, pcALoc); // Set domain matrix locally.
  CHKERRQ(pcRC);
  pcRC = MatDestroy(&pcALoc);
  CHKERRQ(pcRC);

  pcRC = MatAssemblyBegin(pcA, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);
  pcRC = MatAssemblyEnd  (pcA, MAT_FINAL_ASSEMBLY);
  CHKERRQ(pcRC);

  // Use a MatMPI (not a MatIS) if using existing PETSc preconditioners.

  if (opt.useMatMPI) {
    Mat pcTmpA; // Get A as a MatMPI matrix (not a MatIS).
    pcRC = MatConvert(pcA, MATAIJ, MAT_INITIAL_MATRIX, &pcTmpA); // Assemble local parts of A.
    CHKERRQ(pcRC);
    pcRC = MatDestroy(&pcA);
    CHKERRQ(pcRC);
    pcA = pcTmpA; // Replace MatIS with MatMPI.
  }

  // Verbose on demand.

  if (opt.verbose >= 2) {
    pcRC = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_COMMON);
    CHKERRQ(pcRC);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "The matrix A is:\n");
    CHKERRQ(pcRC);
    pcRC = MatView(pcA, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(pcRC);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
    CHKERRQ(pcRC);
  }

  // Debug on demand.

  if (opt.debug) {
    PetscViewer pcView;
    string debugFile = "debug.input.A.MatIS";
    pcRC = createPetscViewer(false, opt.debugMat, PETSC_COMM_WORLD, debugFile, pcView); // MatIS crashes on binary export...
    CHKERRQ(pcRC);
    pcRC = MatView(pcA, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  return 0;
}

int createB(Mat const & pcA, Vec & pcB, options const & opt) {
  // Create and fill B.

  PetscErrorCode pcRC = MatCreateVecs(pcA, &pcB, NULL); // Vector matching the matrix.
  CHKERRQ(pcRC);
  boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
  if (opt.inpFileB.length() == 0) {
    Vec pcX;
    pcRC = MatCreateVecs(pcA, &pcX, NULL); // Vector matching the matrix.
    CHKERRQ(pcRC);
    PetscInt pcDimX = 0;
    pcRC = VecGetSize(pcX, &pcDimX);
    CHKERRQ(pcRC);
    if (petscWorld.rank() == 0) {
      for (PetscInt pcIdx = 0; pcIdx < pcDimX; pcIdx++) {
        pcRC = VecSetValue(pcX, pcIdx, pcIdx+1., INSERT_VALUES); // Fill X with 1., 2., ...
        CHKERRQ(pcRC);
      }
    }
    pcRC = VecAssemblyBegin(pcX);
    CHKERRQ(pcRC);
    pcRC = VecAssemblyEnd  (pcX);
    CHKERRQ(pcRC);

    pcRC = MatMult(pcA, pcX, pcB); // Build B such that the solution X of AX = B is known.
    CHKERRQ(pcRC);
    pcRC = VecDestroy(&pcX);
    CHKERRQ(pcRC);
  }
  else {
    pcRC = VecSet(pcB, 0.);
    CHKERRQ(pcRC);

    if (petscWorld.rank() == 0) {
      ifstream inp(opt.inpFileB);
      if (!inp) {cerr << "Error: can not open " << opt.inpFileB << endl; return 1;}
      do {
        // Skip comments.

        string inpLine; getline(inp, inpLine);
        while (isspace(*inpLine.begin())) inpLine.erase(inpLine.begin()); // Suppress leading white spaces.
        if (inpLine.length() == 0) continue; // Empty line.
        if (inpLine[0] == '%' || inpLine[0] == '#') continue; // Comments skipped, begin reading.

        // Read line by line.

        stringstream inpSS(inpLine);
        PetscInt pcIdx;    inpSS >> pcIdx; if (!inpSS) {cerr << "Error: can not read " << opt.inpFileB << endl; return 1;}
        PetscScalar pcVal; inpSS >> pcVal; if (!inpSS) pcVal = 1.;
        pcRC = VecSetValue(pcB, pcIdx, pcVal, INSERT_VALUES); // Fill B.
        CHKERRQ(pcRC);
      }
      while (inp);
    }
  }
  pcRC = VecAssemblyBegin(pcB);
  CHKERRQ(pcRC);
  pcRC = VecAssemblyEnd  (pcB);
  CHKERRQ(pcRC);

  // Verbose on demand.

  if (opt.verbose >= 2) {
    pcRC = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_COMMON);
    CHKERRQ(pcRC);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "The vector B is:\n");
    CHKERRQ(pcRC);
    pcRC = VecView(pcB, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(pcRC);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
    CHKERRQ(pcRC);
  }

  // Debug on demand.

  if (opt.debug) {
    PetscViewer pcView;
    string debugFile = "debug.input.B";
    pcRC = createPetscViewer(opt.debugBin, opt.debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(pcB, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  return 0;
}

#define SETERRABT(msg) SETERRABORT(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,msg)

int printIterativeGlobalSolveParameters(unsigned int const & nbDOF, unsigned int const & nbSubMat,
                                        unsigned int const & nbNonNullValLoc, bool const & metisDual,
                                        KSP const & pcKSP, PC const & pcPC, options const & opt) {
  // Print parameters of the iterative global solve.

  unsigned int nbNonNullVal = 0;
  boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
  boost::mpi::reduce(petscWorld, nbNonNullValLoc, nbNonNullVal, plus<unsigned int>(), 0);
  unsigned int const nbPart = petscWorld.size();
  string info = "INFO: nb DOFs %u, nb elements %u, nnz coefs %u, nb partitions %u, overlap %u";
  info += (metisDual) ? ", metis dual\n" : ", metis nodal\n";
  PetscErrorCode pcRC = PetscPrintf(PETSC_COMM_WORLD, info.c_str(), nbDOF, nbSubMat, nbNonNullVal, nbPart, opt.addOverlap);
  CHKERRQ(pcRC);
  PetscReal pcRelTol = 0., pcAbsTol = 0.; PetscInt maxIts = 0;
  pcRC = KSPGetTolerances(pcKSP, &pcRelTol, & pcAbsTol, NULL, &maxIts);
  CHKERRQ(pcRC);
  KSPType pcKSPType;
  pcRC = KSPGetType(pcKSP, &pcKSPType);
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, "INFO: %s ksp, eps rel %.1e", pcKSPType, pcRelTol);
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, ", eps abs %.1e, max iterations %d\n", pcAbsTol, maxIts);
  CHKERRQ(pcRC);
  PCType pcPCType = NULL;
  pcRC = PCGetType(pcPC, &pcPCType);
  CHKERRQ(pcRC);
  if (pcPCType && string(pcPCType) == "geneo") {
    // Get the context.

    if (!pcPC) SETERRABT("GenEO preconditioner is invalid");
    geneoContext * gCtx = (geneoContext *) pcPC->data;
    if (!gCtx) {cerr << "Error: shell preconditioner without context" << endl; return 1;}

    // Print GenEO parameters.

    pcRC = PetscPrintf(PETSC_COMM_WORLD, "INFO: %s pc", gCtx->name.c_str());
    CHKERRQ(pcRC);
    if (gCtx->lvl1ORAS) {
      pcRC = PetscPrintf(PETSC_COMM_WORLD, ", optim %.2f", gCtx->optim);
      CHKERRQ(pcRC);
    }
    if (gCtx->effHybrid) {
      pcRC = PetscPrintf(PETSC_COMM_WORLD, ", initial guess");
      CHKERRQ(pcRC);
    }
    PC pcPCL1Loc;
    pcRC = KSPGetPC(gCtx->pcKSPL1Loc, &pcPCL1Loc);
    CHKERRQ(pcRC);
    MatSolverType pcType = NULL;
    pcRC = PCFactorGetMatSolverType(pcPCL1Loc, &pcType);
    CHKERRQ(pcRC);
    if (pcType) {
      string infoL1(pcType);
      infoL1 += ((gCtx->hybrid) ? " proj-fine-space" : " no-proj-fine-space");
      pcRC = PetscPrintf(PETSC_COMM_WORLD, ", L1 %s", infoL1.c_str());
      CHKERRQ(pcRC);
    }
    if (gCtx->lvl2) {
      pcRC = PetscPrintf(PETSC_COMM_WORLD, ", tau %.2f", gCtx->tau);
      CHKERRQ(pcRC);
      if (gCtx->lvl2 >= 2) {
        pcRC = PetscPrintf(PETSC_COMM_WORLD, ", gamma %.2f", gCtx->gamma);
        CHKERRQ(pcRC);
      }
      if (gCtx->offload) {
        pcRC = PetscPrintf(PETSC_COMM_WORLD, ", offload");
        CHKERRQ(pcRC);
      }
      pcRC = PetscPrintf(PETSC_COMM_WORLD, ", L2 %s\n", gCtx->infoL2.c_str());
      CHKERRQ(pcRC);
      if (!opt.shortRes) {
        pcRC = PetscPrintf(PETSC_COMM_WORLD, "INFO: setup - ");
        CHKERRQ(pcRC);
        int estimDimE = 0, estimDimEMin = 0, estimDimEMax = 0, realDimE = 0, realDimEMin = 0, realDimEMax = 0;
        boost::mpi::reduce(petscWorld, gCtx->estimDimELoc, estimDimE, plus<int>(), 0);
        boost::mpi::reduce(petscWorld, gCtx->estimDimELoc, estimDimEMin, boost::mpi::minimum<int>(), 0);
        boost::mpi::reduce(petscWorld, gCtx->estimDimELoc, estimDimEMax, boost::mpi::maximum<int>(), 0);
        boost::mpi::reduce(petscWorld, gCtx->realDimELoc, realDimE, plus<int>(), 0);
        boost::mpi::reduce(petscWorld, gCtx->realDimELoc, realDimEMin, boost::mpi::minimum<int>(), 0);
        boost::mpi::reduce(petscWorld, gCtx->realDimELoc, realDimEMax, boost::mpi::maximum<int>(), 0);
        int nicolaides = 0;
        boost::mpi::reduce(petscWorld, gCtx->nicolaidesLoc, nicolaides, plus<int>(), 0);
        if (!gCtx->noSyl) { // Use Sylvester's law.
          pcRC = PetscPrintf(PETSC_COMM_WORLD, "estim dimE %i (local: min %i, max %i), ", estimDimE, estimDimEMin, estimDimEMax);
          CHKERRQ(pcRC);
        }
        pcRC = PetscPrintf(PETSC_COMM_WORLD, ", real dimE %i (local: min %i, max %i)", realDimE, realDimEMin, realDimEMax);
        CHKERRQ(pcRC);
        pcRC = PetscPrintf(PETSC_COMM_WORLD, ", nicolaides %i\n", nicolaides);
        CHKERRQ(pcRC);
      }
    }
    else {
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
      CHKERRQ(pcRC);
      if (!opt.shortRes) {
        pcRC = PetscPrintf(PETSC_COMM_WORLD, "INFO: setup - none\n");
        CHKERRQ(pcRC);
      }
    }
  }
  else {
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "INFO: %s pc\n", pcPCType);
    CHKERRQ(pcRC);
    if (!opt.shortRes) {
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "INFO: setup - none\n");
      CHKERRQ(pcRC);
    }
  }
  pcRC = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD); // Flush to get readable outputs.
  CHKERRQ(pcRC);

  return 0;
}

string getKSPConvergedReason(KSPConvergedReason const & pcReason) {
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

int printIterativeGlobalSolveResults(KSP const & pcKSP, Mat const & pcA, Vec const & pcX, Vec const & pcB,
                                     bool const & shortRes) {
  // Print results of the iterative global solve.

  KSPConvergedReason pcReason;
  PetscErrorCode pcRC = KSPGetConvergedReason(pcKSP, &pcReason);
  CHKERRQ(pcRC);
  string info = string("INFO: solve - ") + string((pcReason >= 0) ? "converged" : "diverged");
  if (shortRes) {
    pcRC = PetscPrintf(PETSC_COMM_WORLD, info.c_str());
    CHKERRQ(pcRC);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
    CHKERRQ(pcRC);
    pcRC = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD); // Flush to get readable outputs.
    CHKERRQ(pcRC);
    return 0;
  }
  info += " (" + getKSPConvergedReason(pcReason) + ")";
  pcRC = PetscPrintf(PETSC_COMM_WORLD, info.c_str());
  CHKERRQ(pcRC);
  PetscInt pcItNb = 0;
  pcRC = KSPGetIterationNumber(pcKSP, &pcItNb);
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, ", %d iteration(s)", pcItNb);
  CHKERRQ(pcRC);
  PetscReal pcResNorm = 0.;
  pcRC = KSPGetResidualNorm(pcKSP, &pcResNorm);
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, ", residual norm %.10f", pcResNorm);
  CHKERRQ(pcRC);
  Vec pcChkRes;
  pcRC = MatCreateVecs(pcA, &pcChkRes, NULL); // Vector matching the matrix (avoid handling layout).
  CHKERRQ(pcRC);
  pcRC = MatMult(pcA, pcX, pcChkRes); // AX.
  CHKERRQ(pcRC);
  pcRC = VecAXPY(pcChkRes, -1., pcB); // AX - B.
  CHKERRQ(pcRC);
  PetscReal pcChkResNorm;
  pcRC = VecNorm(pcChkRes, NORM_2, &pcChkResNorm);
  CHKERRQ(pcRC);
  pcRC = VecDestroy(&pcChkRes);
  CHKERRQ(pcRC);
  PetscReal pcBNorm;
  pcRC = VecNorm(pcB, NORM_2, &pcBNorm);
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, ", || AX - B || / || B || %.10f", pcChkResNorm/pcBNorm);
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
  CHKERRQ(pcRC);
  pcRC = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD); // Flush to get readable outputs.
  CHKERRQ(pcRC);

  return ((pcReason >= 0) ? 0 : 1);
}

int printIterativeGlobalSolveTiming(bool const & timing, PC const & pcPC,
                                    double const & readInpTime, double const & partDecompTime,
                                    double const & createATime, double const & kspSetUpTime, double const & kspItsTime) {
  if (!timing) return 0;

  // Print timers.

  PetscErrorCode pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, "TIME: read input %.5f s, part / decomp %.5f s", readInpTime, partDecompTime);
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, ", create A %.5f s, solver set up %.5f s", createATime, kspSetUpTime);
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, ", solver iterations %.5f s, solve %.5f s", kspItsTime, kspItsTime + kspSetUpTime);
  CHKERRQ(pcRC);
  pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
  CHKERRQ(pcRC);
  pcRC = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD); // Flush to get readable outputs.
  CHKERRQ(pcRC);
  PCType pcPCType = NULL;
  pcRC = PCGetType(pcPC, &pcPCType);
  CHKERRQ(pcRC);
  if (pcPCType && string(pcPCType) == "geneo") {
    // Get the context.

    if (!pcPC) SETERRABT("GenEO preconditioner is invalid");
    geneoContext * gCtx = (geneoContext *) pcPC->data;
    if (!gCtx) {cerr << "Error: shell preconditioner without context" << endl; return 1;}

    // Print GenEO timers.

    boost::mpi::communicator petscWorld = boost::mpi::communicator(PETSC_COMM_WORLD, boost::mpi::comm_create_kind::comm_attach);
    double lvl1SetupMinvTime = 0.;
    boost::mpi::reduce(petscWorld, gCtx->lvl1SetupMinvTimeLoc, lvl1SetupMinvTime, boost::mpi::maximum<double>(), 0);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "      L1       setup: Minv %.5f s\n", lvl1SetupMinvTime);
    CHKERRQ(pcRC);
    if (gCtx->lvl2) {
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "      L2       setup: ");
      CHKERRQ(pcRC);
      double lvl2SetupSylTime = 0., lvl2SetupEigTime = 0., lvl2SetupZTime = 0., lvl2SetupETime = 0.;
      boost::mpi::reduce(petscWorld, gCtx->lvl2SetupSylTimeLoc, lvl2SetupSylTime, boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl2SetupEigTimeLoc, lvl2SetupEigTime, boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl2SetupZTimeLoc,   lvl2SetupZTime,   boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl2SetupETimeLoc,   lvl2SetupETime,   boost::mpi::maximum<double>(), 0);
      if (!gCtx->noSyl) { // Use Sylvester's law.
        pcRC = PetscPrintf(PETSC_COMM_WORLD, "sylvester %.5f s, ", lvl2SetupSylTime);
        CHKERRQ(pcRC);
      }
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "eigen solve %.5f s", lvl2SetupEigTime);
      CHKERRQ(pcRC);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, ", Z %.5f s", lvl2SetupZTime);
      CHKERRQ(pcRC);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, ", E %.5f s", lvl2SetupETime);
      CHKERRQ(pcRC);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
      CHKERRQ(pcRC);

      pcRC = PetscPrintf(PETSC_COMM_WORLD, "      L2 tau   setup: ");
      CHKERRQ(pcRC);
      double lvl2SetupTauLocTime = 0., lvl2SetupTauSylTime = 0., lvl2SetupTauEigTime = 0.;
      boost::mpi::reduce(petscWorld, gCtx->lvl2SetupTauLocTimeLoc, lvl2SetupTauLocTime, boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl2SetupTauSylTimeLoc, lvl2SetupTauSylTime, boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl2SetupTauEigTimeLoc, lvl2SetupTauEigTime, boost::mpi::maximum<double>(), 0);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "tau   loc %.5f s", lvl2SetupTauLocTime);
      CHKERRQ(pcRC);
      if (!gCtx->noSyl) { // Use Sylvester's law.
        pcRC = PetscPrintf(PETSC_COMM_WORLD, ", sylvester %.5f s", lvl2SetupTauSylTime);
        CHKERRQ(pcRC);
      }
      pcRC = PetscPrintf(PETSC_COMM_WORLD, ", eigen solve %.5f s", lvl2SetupTauEigTime);
      CHKERRQ(pcRC);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
      CHKERRQ(pcRC);
      if (gCtx->lvl2 >= 2) {
        pcRC = PetscPrintf(PETSC_COMM_WORLD, "      L2 gamma setup: ");
        CHKERRQ(pcRC);
        double lvl2SetupGammaLocTime = 0., lvl2SetupGammaSylTime = 0., lvl2SetupGammaEigTime = 0.;
        boost::mpi::reduce(petscWorld, gCtx->lvl2SetupGammaLocTimeLoc, lvl2SetupGammaLocTime, boost::mpi::maximum<double>(), 0);
        boost::mpi::reduce(petscWorld, gCtx->lvl2SetupGammaSylTimeLoc, lvl2SetupGammaSylTime, boost::mpi::maximum<double>(), 0);
        boost::mpi::reduce(petscWorld, gCtx->lvl2SetupGammaEigTimeLoc, lvl2SetupGammaEigTime, boost::mpi::maximum<double>(), 0);
        pcRC = PetscPrintf(PETSC_COMM_WORLD, "gamma loc %.5f s", lvl2SetupGammaLocTime);
        CHKERRQ(pcRC);
        if (!gCtx->noSyl) { // Use Sylvester's law.
          pcRC = PetscPrintf(PETSC_COMM_WORLD, ", sylvester %.5f s", lvl2SetupGammaSylTime);
          CHKERRQ(pcRC);
        }
        pcRC = PetscPrintf(PETSC_COMM_WORLD, ", eigen solve %.5f s", lvl2SetupGammaEigTime);
        CHKERRQ(pcRC);
        pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
        CHKERRQ(pcRC);
      }
    }
    double lvl1ApplyTime = 0., lvl1ApplyScatterTime = 0., lvl1ApplyMinvTime = 0., lvl1ApplyGatherTime = 0.;
    boost::mpi::reduce(petscWorld, gCtx->lvl1ApplyTimeLoc,        lvl1ApplyTime,        boost::mpi::maximum<double>(), 0);
    boost::mpi::reduce(petscWorld, gCtx->lvl1ApplyScatterTimeLoc, lvl1ApplyScatterTime, boost::mpi::maximum<double>(), 0);
    boost::mpi::reduce(petscWorld, gCtx->lvl1ApplyMinvTimeLoc,    lvl1ApplyMinvTime,    boost::mpi::maximum<double>(), 0);
    boost::mpi::reduce(petscWorld, gCtx->lvl1ApplyGatherTimeLoc,  lvl1ApplyGatherTime,  boost::mpi::maximum<double>(), 0);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "      L1       solve: apply %.5f s - ", lvl1ApplyTime);
    CHKERRQ(pcRC);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "scatter %.5f s, Minv %.5f s, gather %.5f s", lvl1ApplyScatterTime, lvl1ApplyMinvTime, lvl1ApplyGatherTime);
    CHKERRQ(pcRC);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
    CHKERRQ(pcRC);
    if (gCtx->hybrid) {
      double lvl1ApplyPrjFSTime = 0., lvl1ApplyPrjFSZtTime = 0., lvl1ApplyPrjFSEinvTime = 0., lvl1ApplyPrjFSZTime = 0.;
      boost::mpi::reduce(petscWorld, gCtx->lvl1ApplyPrjFSTimeLoc,     lvl1ApplyPrjFSTime,     boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl1ApplyPrjFSZtTimeLoc,   lvl1ApplyPrjFSZtTime,   boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl1ApplyPrjFSEinvTimeLoc, lvl1ApplyPrjFSEinvTime, boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl1ApplyPrjFSZTimeLoc,    lvl1ApplyPrjFSZTime,    boost::mpi::maximum<double>(), 0);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "      L1       solve: prjFS %.5f s - ", lvl1ApplyPrjFSTime);
      CHKERRQ(pcRC);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "Zt %.5f s, Einv %.5f s, Z %.5f s", lvl1ApplyPrjFSZtTime, lvl1ApplyPrjFSEinvTime, lvl1ApplyPrjFSZTime);
      CHKERRQ(pcRC);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
      CHKERRQ(pcRC);
    }
    if (gCtx->lvl2) {
      double lvl2ApplyTime = 0., lvl2ApplyZtTime = 0., lvl2ApplyEinvTime = 0., lvl2ApplyZTime = 0.;
      boost::mpi::reduce(petscWorld, gCtx->lvl2ApplyTimeLoc,     lvl2ApplyTime,     boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl2ApplyZtTimeLoc,   lvl2ApplyZtTime,   boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl2ApplyEinvTimeLoc, lvl2ApplyEinvTime, boost::mpi::maximum<double>(), 0);
      boost::mpi::reduce(petscWorld, gCtx->lvl2ApplyZTimeLoc,    lvl2ApplyZTime,    boost::mpi::maximum<double>(), 0);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "      L2       solve: apply %.5f s - ", lvl2ApplyTime);
      CHKERRQ(pcRC);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "Zt %.5f s, Einv %.5f s, Z %.5f s", lvl2ApplyZtTime, lvl2ApplyEinvTime, lvl2ApplyZTime);
      CHKERRQ(pcRC);
      pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
      CHKERRQ(pcRC);
    }
    pcRC = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD); // Flush to get readable outputs.
    CHKERRQ(pcRC);
  }

  return 0;
}

int iterativeGlobalSolve(KSP const & pcKSP, Mat const & pcA, Vec const & pcB, Vec & pcX, PC const & pcPC,
                         unsigned int const & nbDOF, unsigned int const & nbSubMat, unsigned int const & nbNonNullValLoc,
                         double const & readInpTime, double const & partDecompTime, double const & createATime, double const & kspSetUpTime,
                         options const & opt) {
  // Solve.

  auto start = chrono::high_resolution_clock::now();
  PetscErrorCode pcRC = KSPSolve(pcKSP, pcB, pcX);
  CHKERRQ(pcRC);
  auto stop = chrono::high_resolution_clock::now();

  // Verbose on demand.

  if (opt.verbose >= 1) {
    pcRC = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_COMMON);
    CHKERRQ(pcRC);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "The solution X is:\n");
    CHKERRQ(pcRC);
    pcRC = VecView(pcX, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(pcRC);
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "\n");
    CHKERRQ(pcRC);
  }

  // Debug on demand.

  if (opt.debug) {
    PetscViewer pcView;
    string debugFile = "debug.output.X";
    pcRC = createPetscViewer(opt.debugBin, opt.debugMat, PETSC_COMM_WORLD, debugFile, pcView);
    CHKERRQ(pcRC);
    pcRC = VecView(pcX, pcView);
    CHKERRQ(pcRC);
    pcRC = PetscViewerDestroy(&pcView);
    CHKERRQ(pcRC);
  }

  // Print parameters and results.

  int rc = printIterativeGlobalSolveParameters(nbDOF, nbSubMat, nbNonNullValLoc, opt.metisDual, pcKSP, pcPC, opt);
  if (rc != 0) {cerr << "Error: print iterative global solve parameters KO" << endl; return 1;}
  rc = printIterativeGlobalSolveResults(pcKSP, pcA, pcX, pcB, opt.shortRes);
  if (rc != 0) {cerr << "Error: print iterative global solve results KO" << endl; return 1;}
  auto kspItsTime = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
  rc = printIterativeGlobalSolveTiming(opt.timing, pcPC, readInpTime, partDecompTime, createATime, kspSetUpTime, kspItsTime);
  if (rc != 0) {cerr << "Error: print iterative global solve timing KO" << endl; return 1;}

  return 0;
}

int solve(unsigned int const & nbDOF, unsigned int const & nbSubMat,
          unsigned int const & nbDOFLoc, unsigned int const & nbSubMatLoc,
          vector<unsigned int> const & subMatPtrLoc, vector<unsigned int> const & subMatIdxLoc,
          vector<vector<PetscScalar>> const & subMatLoc,
          set<unsigned int> const & dofIdxDomLoc, vector<unsigned int> const & dofIdxMultLoc,
          vector<vector<unsigned int>> const & intersectLoc,
          options const & opt, double const & readInpTime, double const & partDecompTime) {
  // Create local to global mapping.

  vector<PetscInt> pcIdxDomLoc;
  pcIdxDomLoc.reserve(nbDOFLoc);
  for (auto idx = dofIdxDomLoc.cbegin(); idx != dofIdxDomLoc.cend(); idx++) pcIdxDomLoc.push_back(*idx);

  PetscErrorCode pcRC;
  ISLocalToGlobalMapping pcMap;
  pcRC = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, nbDOFLoc, pcIdxDomLoc.data(), PETSC_COPY_VALUES, &pcMap);
  CHKERRQ(pcRC);

  // Create A, B and a shell ASM preconditioner.

  auto start = chrono::high_resolution_clock::now();
  Mat pcA;
  unsigned int nbNonNullValLoc = 0;
  int rc = createA(nbDOFLoc, nbSubMatLoc, subMatPtrLoc, subMatIdxLoc, subMatLoc, dofIdxDomLoc,
                   nbDOF, pcMap, pcA, nbNonNullValLoc, opt);
  if (rc != 0) {cerr << "Error: createA KO" << endl; return 1;}
  pcRC = MatSetOptionsPrefix(pcA, "A");
  CHKERRQ(pcRC);
  pcRC = MatSetFromOptions(pcA);
  CHKERRQ(pcRC);
  auto stop = chrono::high_resolution_clock::now();
  double createATime = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;

  Vec pcB;
  rc = createB(pcA, pcB, opt);
  if (rc != 0) {cerr << "Error: createB KO" << endl; return 1;}
  pcRC = VecSetOptionsPrefix(pcB, "B");
  CHKERRQ(pcRC);
  pcRC = VecSetFromOptions(pcB);
  CHKERRQ(pcRC);

  Vec pcX;
  pcRC = VecDuplicate(pcB, &pcX); // Get the same layout than B.
  CHKERRQ(pcRC);

  KSP pcKSP;
  pcRC = KSPCreate(PETSC_COMM_WORLD, &pcKSP); // Global solve (shared by all MPI proc).
  CHKERRQ(pcRC);
  pcRC = PCRegister("geneo", createGenEOPC);
  CHKERRQ(pcRC);
  PC pcPC;
  pcRC = KSPGetPC(pcKSP, &pcPC);
  CHKERRQ(pcRC);
  pcRC = PCSetFromOptions(pcPC); // Just before setup to override default options.
  CHKERRQ(pcRC);
  PCType pcPCType = NULL;
  pcRC = PCGetType(pcPC, &pcPCType);
  CHKERRQ(pcRC);
  if (pcPCType && string(pcPCType) == "geneo") {
    auto *dofIdxDomLoc_vector = new vector<unsigned int>;
    dofIdxDomLoc_vector->reserve(dofIdxDomLoc.size());
    for (auto idx = dofIdxDomLoc.cbegin(); idx != dofIdxDomLoc.cend(); idx++)
      dofIdxDomLoc_vector->push_back(*idx);
    pcRC = initGenEOPC(pcPC, nbDOF, nbDOFLoc, pcMap, pcA, pcB, pcX, dofIdxDomLoc_vector, &dofIdxMultLoc, &intersectLoc);
    CHKERRQ(pcRC);
    pcRC = KSPSetInitialGuessNonzero(pcKSP, PETSC_TRUE);
    CHKERRQ(pcRC);
  }

  // Set up solver and solve.

  pcRC = KSPSetOperators(pcKSP, pcA, pcA); // Set A as operator.
  CHKERRQ(pcRC);
  pcRC = KSPSetFromOptions(pcKSP); // Just before setup to override default options.
  CHKERRQ(pcRC);
  if (opt.debug) {
    pcRC = KSPSetResidualHistory(pcKSP, NULL, PETSC_DECIDE, PETSC_TRUE);
    CHKERRQ(pcRC);
  }

  start = chrono::high_resolution_clock::now();
  pcRC = KSPSetUp(pcKSP);
  CHKERRQ(pcRC);
  stop = chrono::high_resolution_clock::now();
  double kspSetUpTime = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
  rc = iterativeGlobalSolve(pcKSP, pcA, pcB, pcX, pcPC, nbDOF, nbSubMat, nbNonNullValLoc,
                            readInpTime, partDecompTime, createATime, kspSetUpTime, opt);
  if (rc != 0) {cerr << "Error: iterative global solve KO" << endl; return 1;}
  if (opt.debug) {
    PetscInt pcItNb = 0;
    PetscReal * pcResNorm = NULL;
    pcRC = KSPGetResidualHistory(pcKSP, &pcResNorm, &pcItNb);
    CHKERRQ(pcRC);
    ofstream debug("debug.apply.residual.log");
    for (PetscInt pcIdx = 0; pcIdx < pcItNb; pcIdx++) debug << "Iteration " << pcIdx << ": residual " << pcResNorm[pcIdx] << endl;
  }

  // Clean.

  pcRC = KSPDestroy(&pcKSP);
  CHKERRQ(pcRC);
  pcRC = VecDestroy(&pcB);
  CHKERRQ(pcRC);
  pcRC = VecDestroy(&pcX);
  CHKERRQ(pcRC);
  pcRC = MatDestroy(&pcA);
  CHKERRQ(pcRC);
  pcRC = ISLocalToGlobalMappingDestroy(&pcMap);
  CHKERRQ(pcRC);

  return 0;
}

int checkArguments(int argc, char ** argv, options & opt) {
  opt.inpFileA = ""; opt.inpEps = 0.0001; opt.inpLibA = ""; opt.inpLibArg = "";
  opt.inpFileB = "";
  opt.metisDual = true;
  opt.useMatMPI = true;
  opt.addOverlap = 0;
  opt.debug = opt.debugBin = opt.debugMat = false;
  opt.verbose = 0;
  opt.timing = false; opt.shortRes = false; opt.cmdLine = false;

  // Keep track of user options.

  for (int a = 0; argv && a < argc; a++) opt.userCmdLine << argv[a] << " ";

  // Checking options.

  for (int a = 0; argv && a < argc; a++) {
    string clo = argv[a]; // Command line option
    if (clo == "--help") return -1;
    if (clo == "--inpFileA") {
      a++; if (a >= argc) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      opt.inpFileA = argv[a];
      argv[a] = argv[a-1] = NULL; // Clear option(s).
    }
    if (clo == "--inpEps") {
      a++; if (a >= argc) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      stringstream inpEpsilon(argv[a]);
      inpEpsilon >> opt.inpEps;
      if (!inpEpsilon) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      argv[a] = argv[a-1] = NULL; // Clear option(s).
    }
    if (clo == "--inpLibA") {
      a++; if (a >= argc) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      stringstream inpLibrary(argv[a]);
      inpLibrary >> opt.inpLibA;
      if (!inpLibrary) {cerr << "Error: invalid command line, " << clo << endl; return 1;}

      a++; if (a >= argc) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      stringstream inpLibraryArg(argv[a]);
      inpLibraryArg >> opt.inpLibArg;
      if (!inpLibraryArg) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      argv[a] = argv[a-1] = argv[a-2] = NULL; // Clear option(s).
    }
    if (clo == "--inpFileB") {
      a++; if (a >= argc) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      opt.inpFileB = argv[a];
      argv[a] = argv[a-1] = NULL; // Clear option(s).
    }
    if (clo == "--metisDual") {
      opt.metisDual = true;
      argv[a] = NULL; // Clear option(s).
    }
    if (clo == "--metisNodal") {
      opt.metisDual = false;
      argv[a] = NULL; // Clear option(s).
    }
    if (clo == "geneo") { // Do NOT clear option !
      opt.useMatMPI = false;
    }
    if (clo == "--addOverlap") {
      a++; if (a >= argc) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      stringstream addOverlap(argv[a]);
      addOverlap >> opt.addOverlap;
      if (!addOverlap) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      argv[a] = argv[a-1] = NULL; // Clear option(s).
    }
    if (clo == "--debug") {
      opt.debug = true;
      a++; if (a >= argc) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      stringstream debugFile(argv[a]);
      if (debugFile.str() == "bin") opt.debugBin = true;
      if (debugFile.str() == "mat") opt.debugMat = true;
      argv[a] = argv[a-1] = NULL; // Clear option(s).
    }
    if (clo == "--verbose") {
      a++; if (a >= argc) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      stringstream verboseLevel(argv[a]);
      verboseLevel >> opt.verbose;
      if (!verboseLevel) {cerr << "Error: invalid command line, " << clo << endl; return 1;}
      argv[a] = argv[a-1] = NULL; // Clear option(s).
    }
    if (clo == "--timing") {
      opt.timing = true;
      argv[a] = NULL; // Clear option(s).
    }
    if (clo == "--shortRes") {
      opt.shortRes = true;
      argv[a] = NULL; // Clear option(s).
    }
    if (clo == "--cmdLine") {
      opt.cmdLine = true;
      argv[a] = NULL; // Clear option(s).
    }
  }

  if (opt.inpFileA.length() == 0 && opt.inpLibA.length() == 0) {cerr << "Error: no input" << endl; return 1;}
  if (opt.inpFileA.length() != 0 && opt.inpLibA.length() != 0) {cerr << "Error: several input" << endl; return 1;}

  return 0;
}

void usage() {
  cerr << endl;
  cerr << "usage: geneo4PETSc is an implementation of the GenEO preconditioner with PETSc and SLEPc" << endl;
  cerr << "" << endl;
  cerr << "  --help,         print help related to geneo4PETSc" << endl;
  cerr << "  --inpFileA F,   input file F describing the A matrix (mandatory)" << endl;
  cerr << "                  - a unique ID (number) is attributed to each degree of freedom of the problem (described by A)" << endl;
  cerr << "                  - each line of the input file stands for one element described like:" << endl;
  cerr << "                    - a list of n degrees of freedom (mandatory)" << endl;
  cerr << "                    - a \"-\" followed by a dense row ordered nxn matrix (optional)" << endl;
  cerr << "                      note: if the dense matrix is not specified, a default one is built with --inpEps" << endl;
  cerr << "                  you can pass arguments to the A matrix at the command line using prefix -A_" << endl;
  cerr << "  --inpEps E,     epsilon used to tune the elementary matrix (defaults to 0.0001)" << endl;
  cerr << "                  the default nDOF x nDOF elementary matrix is:" << endl;
  cerr << "                  | 1.+eps,  alpha,  alpha,  alpha, ... |" << endl;
  cerr << "                  |  alpha, 1.+eps,  alpha,  alpha, ... |" << endl;
  cerr << "                  |  alpha,  alpha, 1.+eps,  alpha, ... | where alpha = -1./(nDOF-1)" << endl;
  cerr << "                  |  alpha,  alpha,  alpha, 1.+eps, ... |" << endl;
  cerr << "                  |    ...,    ...,    ...,    ..., ... |" << endl;
  cerr << "  --inpLibA L A,  input file describing the A matrix provided as a library (.so)" << endl;
  cerr << "                  L: path to the library /path/to/libinput.so" << endl;
  cerr << "                  A: arguments to pass to the library (string)" << endl;
  cerr << "                     for geneo4PETSc, A must be one token, but for L, A may be a list of (sub) tokens" << endl;
  cerr << "                     the geneo4PETSc token will be the concatenation of all L sub tokens with #" << endl;
  cerr << "                     example: --inpLibA /path/to/lib --param1=1#--param2=2#--param3=3" << endl;
  cerr << "                  the library must provide and implement the following function" << endl;
  cerr << "                    /*" << endl;
  cerr << "                     * getInput: provide inputs (elements and associated matrices) for domain decomposition" << endl;
  cerr << "                     *           each element is considered to be a set of n DOFs" << endl;
  cerr << "                     * input parameter:" << endl;
  cerr << "                     *   - args: command line arguments passed to L needed to create inputs (if any)" << endl;
  cerr << "                     * output parameters:" << endl;
  cerr << "                     *   - nbElem: number of elements" << endl;
  cerr << "                     *   - nbNode: number of nodes" << endl;
  cerr << "                     *   - elemPtr: list of element pointers (CSR format)" << endl;
  cerr << "                     *   - elemIdx: list of element indices (CSR format)" << endl;
  cerr << "                     *   - elemSubMat: list of matrice associated to each element" << endl;
  cerr << "                     *                 each matrice is a dense row ordered nxn matrix (std::vector<double>)" << endl;
  cerr << "                     * return:" << endl;
  cerr << "                     *   - integer as return code (0 if OK, error code otherwise)" << endl;
  cerr << "                     * linking:" << endl;
  cerr << "                     *   - use extern \"C\" to get a C style decoration of symbols" << endl;
  cerr << "                     */ " << endl;
  cerr << "                    int getInput(std::string const & args," << endl;
  cerr << "                                 unsigned int & nbElem, unsigned int & nbNode," << endl;
  cerr << "                                 std::vector<unsigned int> & elemPtr, std::vector<unsigned int> & elemIdx," << endl;
  cerr << "                                 std::vector<std::vector<double>> & elemSubMat)" << endl;
  cerr << "                  you can pass arguments to the A matrix at the command line using prefix -A_" << endl;
  cerr << "  --inpFileB F,   input file F describing the B vector" << endl;
  cerr << "                  - a unique ID (number) is attributed to each degree of freedom of the problem (described by A)" << endl;
  cerr << "                  - each line of the input file stands for one degree of freedom described like:" << endl;
  cerr << "                    - one degree of freedom (mandatory)" << endl;
  cerr << "                    - one value (optional)" << endl;
  cerr << "                      note: if the value is not specified, the default one is 1." << endl;
  cerr << "                  you can pass arguments to the B vector at the command line using prefix -B_" << endl;
  cerr << "  --metisDual,    use dual approach to partition input data with metis (partition according to elements)" << endl;
  cerr << "                  each element belongs to one partition only" << endl;
  cerr << "  --metisNodal,   use nodal approach to partition input data with metis (partition according to nodes)" << endl;
  cerr << "                  each element belongs to a partition if one of its nodes does" << endl;
  cerr << "                  each element can belong to several partitions" << endl;
  cerr << "  --addOverlap L, add L layers of overlap at each domain borders" << endl;
  cerr << "  --debug F,      create debug files for steps before and after solve (metis, A, B, X)" << endl;
  cerr << "                  F = log (ASCII file), bin (binary file) or mat (matlab file)" << endl;
  cerr << "  --verbose V,    dump A, B and X" << endl;
  cerr << "                  V = 1: dumps X" << endl;
  cerr << "                  V = 2: dumps A, B and X" << endl;
  cerr << "  --timing,       print timing" << endl;
  cerr << "  --shortRes,     print short result status (makes output stable for test suite checks)" << endl;
  cerr << "  --cmdLine,      print command line at the end of the log" << endl;
  cerr << usageGenEO(false);
}

int main(int argc, char ** argv) {
  auto start = chrono::high_resolution_clock::now();

  // Check command line arguments.

  options opt;
  int rc = checkArguments(argc, argv, opt);
  if (rc != 0) {usage(); return (rc == -1) ? 0 : 1;} // Return code 0 if asking for help.

  // Partition (done by the master, then, sent by master to slaves).

  boost::mpi::environment env;
  unsigned int nbNode = 0, nbElem = 0;
  unsigned int nbNodeLoc = 0, nbElemLoc = 0;
  vector<unsigned int> elemPtrLoc, elemIdxLoc;
  vector<vector<PetscScalar>> elemSubMatLoc; // Sub matrix for each element of the local domain.
  set<unsigned int> nodeIdxDomLoc; // Set of (global) indices of nodes standing for local domains.
  vector<unsigned int> nodeIdxMultLoc; // Multiplicity of nodes of the local domain.
  vector<vector<unsigned int>> intersectLoc; // Intersections with other domains.
  double readInpTime = 0., partDecompTime = 0.;
  rc = partitionAndDecompose(nbNode, nbElem, nbNodeLoc, nbElemLoc,
                             elemPtrLoc, elemIdxLoc, elemSubMatLoc, nodeIdxDomLoc, nodeIdxMultLoc,
                             intersectLoc, readInpTime, partDecompTime, opt);
  if (rc != 0) {cerr << "Error: partition and decompose KO" << endl; return MPI_Abort(MPI_COMM_WORLD, 1);}

  // Suppress empty domain(s) from "MPI world" if needed.

  boost::mpi::communicator world, petscWorld;
  petscWorld = world.split((nbNodeLoc > 0) ? 1 : 0); // Suppress empty domain(s) (don't crash when procs > data size).
  if (nbNodeLoc == 0) return 0; // Nothing to do.
  PETSC_COMM_WORLD = MPI_Comm(petscWorld); // Suppress empty domain(s) form "PETSc world".

  // Solve with PETSc and SLEPc.

  PetscErrorCode pcRC = PetscInitialize(&argc, &argv, NULL, "");
  CHKERRQ(pcRC);
  pcRC = SlepcInitialize(&argc, &argv, NULL, "");
  CHKERRQ(pcRC);

  rc = solve(nbNode, nbElem, nbNodeLoc, nbElemLoc,
             elemPtrLoc, elemIdxLoc, elemSubMatLoc, nodeIdxDomLoc, nodeIdxMultLoc, intersectLoc,
             opt, readInpTime, partDecompTime);
  if (rc != 0) {cerr << "Error: solve KO" << endl; return MPI_Abort(MPI_COMM_WORLD, 1);}

  auto stop = chrono::high_resolution_clock::now();
  if (opt.timing) {
    auto totTime = chrono::duration_cast<chrono::milliseconds>(stop - start).count()/1000.;
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "TIME: total time %.5f s\n", totTime);
    CHKERRQ(pcRC);
  }
  if (opt.cmdLine) {
    stringstream size; size << world.size();
    string userCmdLine = "mpirun -n " + size.str() + " " + opt.userCmdLine.str();
    pcRC = PetscPrintf(PETSC_COMM_WORLD, "\nCMD: %s\n", userCmdLine.c_str());
    CHKERRQ(pcRC);
  }

  pcRC = SlepcFinalize();
  CHKERRQ(pcRC);
  pcRC = PetscFinalize();
  CHKERRQ(pcRC);

  return 0;
}
