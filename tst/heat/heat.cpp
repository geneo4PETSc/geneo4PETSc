/*
 * This piece of code is designed to generate a heat problem.
 * A dirichlet condition is added to get an invertible matrix.
 *
 * Equation: dT/dt - lbd*div(grad(T)) = 0
 * Explicit time scheme: T^(n+1)/dt - lbd*div(grad(T^(n+1))) = s with a known source s = T^n/dt
 * Variational formulation: int[omega](uv/dt + lbd*grad(u).grad(v))
 */

#include <laplacianServices.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <tuple>
#include <map> // multimap.
#include <cmath> // sqrt, cbrt.

using namespace std;

int getInertia(bool const & bc, vector<double> & elemMat) {
  elemMat.clear();

  /*              1          */
  /*              .          */
  /*             / \         */
  /*            /   \ 0      */
  /*  phi_i  --o  o  o--o--  */
  /*              i  j       */

  /*                 1       */
  /*                 .       */
  /*                / \      */
  /*               /   \ 0   */
  /*  phi_j  --o--o  o  o--  */
  /*              i  j       */

  //                i     j
  //            | i_ii  i_ij | i
  //  inertia = |            |
  //            | i_ji  i_jj | j
  //
  //  distance(i, j) = d = 1.
  //
  //  i_ii = int_[i,j](phi_i.phi_i) = int_[0,1]((1-x)(1-x)) = i_jj (area under the curve)
  //  i_ij = int_[i,j](phi_i.phi_j) = int_[0,1]((1-x)   x ) = i_ji
  //  i_ji = int_[i,j](phi_j.phi_i) = int_[0,1](   x (1-x)) = 1./6.
  //  i_jj = int_[i,j](phi_j.phi_j) = int_[0,1](   x    x ) = 1./3.

  if (!bc) { // No boundary condition.
    elemMat.push_back(1./3.); elemMat.push_back(1./6.);
    elemMat.push_back(1./6.); elemMat.push_back(1./3.);
  }
  else { // Dirichlet boundary condition.
    elemMat.push_back(1./3.); // Ignore ghost point (1st DOF) "out of the matrix" : account only for the contribution of the 2d DOF.
  }

  return 0;
}

int addElement(int const & id1, int const & id2, set<int> & nSet,
               unsigned int & nbElem, vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx,
               vector<vector<double>> & elemSubMat, double const & inpEps,
               string const & kappaInterp, double const & alpha, double const & beta,
               double const & x, double const & y, double const & z,
               double const & lbd, double const & dt,
               bool const & debug, fstream & dbg) {
  if (id1 < 0) {cerr << "Error: invalid ID" << endl; return 1;} // Not allowed, only id2 can be < 0 (boundary condition).

  if (debug) dbg << id1 << " " << id2 << endl;

  // Laplacian.

  vector<double> elemLaplacianMat;
  if (id1 >= 0 && id2 >= 0) {
    nSet.insert(id1); nSet.insert(id2);
    unsigned int ePtrLast = elemPtr[elemPtr.size() - 1]; elemPtr.push_back(ePtrLast + 2);
    elemIdx.push_back(id1); elemIdx.push_back(id2);
    int rc = getLaplacian(inpEps, false, kappaInterp, alpha, beta, x, y, z, elemLaplacianMat);
    if (rc != 0) {cerr << "Error: invalid laplacian" << endl; return 1;}
  }
  else if (id1 >= 0 && id2 < 0) { // Dirichlet boundary condition.
    nSet.insert(id1);
    unsigned int ePtrLast = elemPtr[elemPtr.size() - 1]; elemPtr.push_back(ePtrLast + 1);
    elemIdx.push_back(id1);
    int rc = getLaplacian(inpEps, true, kappaInterp, alpha, beta, x, y, z, elemLaplacianMat);
    if (rc != 0) {cerr << "Error: invalid laplacian" << endl; return 1;}
  }

  // Inertia.

  vector<double> elemInertiaMat;
  if (id1 >= 0 && id2 >= 0) {
    int rc = getInertia(false, elemInertiaMat);
    if (rc != 0) {cerr << "Error: invalid inertia" << endl; return 1;}
  }
  else {
    int rc = getInertia(true, elemInertiaMat);
    if (rc != 0) {cerr << "Error: invalid inertia" << endl; return 1;}
  }

  // Heat = Laplacian + Inertia.

  if (elemLaplacianMat.size() != elemInertiaMat.size()) {cerr << "Error: inconsistent laplacian - inertia" << endl; return 1;}
  vector<double> elemHeatMat;
  for (unsigned int i = 0; i < elemLaplacianMat.size(); i++) elemHeatMat.push_back(lbd*elemLaplacianMat[i] + elemInertiaMat[i]/dt);
  elemSubMat.push_back(elemHeatMat);
  nbElem++;

  return 0;
}

int getIndex(int const & i, int const & j, int const & k, int const & Ni, int const & Nj) {
  return i + Ni*j + Ni*Nj*k;
}

extern "C" {
  int getInput(string const & args,
               unsigned int & nbElem, unsigned int & nbNode,
               vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx, vector<vector<double>> & elemSubMat) {
    // Check arguments.

    int size = 4, weakScaling = 1, dim = 3; double inpEps = 0.0001;
    double kappaMax = 1.; string kappaInterp;
    double lbd = 1., dt = 0.1;
    bool verbose = false, debug = false; fstream dbg;

    if (debug) dbg << "# args: " << args << endl << endl;
    stringstream ssArgs(args);
    while (ssArgs) {
      string opt; ssArgs >> opt; // Command line option
      if (opt == "--size") {
        ssArgs >> size;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--weakScaling") {
        ssArgs >> weakScaling;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--dim") {
        ssArgs >> dim;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
        if (dim != 1 && dim != 2 && dim != 3) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--inpEps") {
        ssArgs >> inpEps;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--kappa") {
        ssArgs >> kappaMax;
        if (!ssArgs || kappaMax < 1.) {cerr << "Error: invalid command line" << endl; return 1;}
        ssArgs >> kappaInterp;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
        bool interpOK = (kappaInterp != "quad" && kappaInterp != "lin" && kappaInterp != "minmax") ? 0 : 1;
        if (!interpOK) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--lbd") {
        ssArgs >> lbd;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--dt") {
        ssArgs >> dt;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--debug") {debug = true; dbg.open("debug.inp", ios::out);}
      if (opt == "--verbose") {verbose = true; cout << "getInput arguments: " << args << endl;}
    }

    // Heat size and dimensions.

    int heatSize = 0;
    if (dim == 1) heatSize =                size*weakScaling;
    if (dim == 2) heatSize = sqrt(     size*size*weakScaling);
    if (dim == 3) heatSize = cbrt(size*size*size*weakScaling);
    if (debug) dbg << "# heat size: " << heatSize << endl << endl;

    int dim1D = 0, dim2D = 0, dim3D = 0;
    if (dim == 1) {dim1D = heatSize; dim2D =        1; dim3D =        1;}
    if (dim == 2) {dim1D = heatSize; dim2D = heatSize; dim3D =        1;}
    if (dim == 3) {dim1D = heatSize; dim2D = heatSize; dim3D = heatSize;}

    // Initialize kappa parameters.

    double alpha = 0.; double beta = 1.;
    int rc = initLaplacian(heatSize, kappaInterp, kappaMax, alpha, beta);
    if (rc != 0) {cerr << "Error: initialize laplacian KO" << endl; return 1;}

    // Heat: laplacian + k * inertia.

    elemPtr.push_back(0);
    multimap<int, tuple<int, int>> elems; // Track which 1D elements have already been processed.
    set<int> nSet;
    for (int d3 = 0; d3 < dim3D; d3++) {
      for (int d2 = 0; d2 < dim2D; d2++) {
        for (int d1 = 0; d1 < dim1D; d1++) {
          int centralIdx = getIndex(d1, d2, d3, dim1D, dim2D); // Central point.

          // Get neighbors (of the central point) according to each direction.

          for (int nd = 1; nd <= 3; nd++) { // Neighbor dimensions.
            vector<int> neighborOffset = {-1, 1};
            for (auto no = neighborOffset.cbegin(); no != neighborOffset.cend(); no++) { // Neighbor offset.
              int nd1 = d1; if (nd == 1) nd1 += *no;
              int nd2 = d2; if (nd == 2) nd2 += *no;
              int nd3 = d3; if (nd == 3) nd3 += *no;

              // Add boundary condition.

              if (nd1 >= dim1D || nd2 >= dim2D || nd3 >= dim3D) continue; // Skip elements out of the domain.
              if (nd1 <      0 || nd2 <      0 || nd3 <      0) {
                bool addBC = false; // Add boundary condition.
                if (dim == 1 && nd == 1 && nd1 == -1) addBC = true;
                if (dim == 2 && nd == 2 && nd2 == -1) addBC = true;
                if (dim == 3 && nd == 3 && nd3 == -1) addBC = true;
                if (addBC) {
                  int rc = addElement(centralIdx, -1, nSet, nbElem, elemPtr, elemIdx, elemSubMat,
                                      inpEps, kappaInterp, alpha, beta, (double) d1, (double) d2, (double) d3, lbd, dt,
                                      debug, dbg);
                  if (rc != 0) {cerr << "Error: invalid element" << endl; return 1;}
                }
                continue; // Skip elements out of the domain.
              }

              // Add element if not already done.

              int neighborIdx = getIndex(nd1, nd2, nd3, dim1D, dim2D); // Neighbor point.
              tuple<int, int> elem(centralIdx, neighborIdx); // 1D element that connects central and neighbor points.

              int key = centralIdx + neighborIdx;
              auto low = elems.lower_bound(key);
              auto upr = elems.upper_bound(key);
              bool addElem = true;
              for (auto it = low; it != upr; it++) {
                tuple<int, int> e = it->second;
                bool ok1 = (get<0>(e) == get<0>(elem) && get<1>(e) == get<1>(elem)) ? true : false;
                bool ok2 = (get<0>(e) == get<1>(elem) && get<1>(e) == get<0>(elem)) ? true : false;
                if (ok1 || ok2) { // 1D elements are not oriented: a->b == b->a.
                  addElem = false; break;
                }
              }
              if (addElem) { // Add element if not already done.
                int rc = addElement(centralIdx, neighborIdx, nSet, nbElem, elemPtr, elemIdx, elemSubMat,
                                    inpEps, kappaInterp, alpha, beta, (double) d1, (double) d2, (double) d3, lbd, dt,
                                    debug, dbg);
                if (rc != 0) {cerr << "Error: invalid element" << endl; return 1;}
                elems.insert(make_pair(key, elem));
              }
            }
          }
        }
      }
    }
    nbNode = nSet.size();

    if (verbose) cout << "getInput: nbNode " << nbNode << ", nbElem " << nbElem << endl;

    return 0;
  }
}
