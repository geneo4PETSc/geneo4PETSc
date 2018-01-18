/*
 * This piece of code is designed to generate a laplacian.
 * A dirichlet condition is added to get an invertible matrix.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <tuple>
#include <map> // multimap.
#include <cmath> // sqrt, cbrt.
#include <algorithm> // transform.

using namespace std;

double computeKappa(string const & kappaInterp, double const & alpha, double const & x, double const & beta) {
  double kappa = 1.;

  if      (kappaInterp == "quad")   kappa = alpha*x*x + beta; // kappa(x) = alpha*x^2 + beta.
  else if (kappaInterp == "lin")    kappa = alpha*x   + beta; // kappa(x) = alpha*x   + beta.
  else if (kappaInterp == "minmax") {
    if (x >=    beta) kappa = alpha; // kappa is a alpha layer.
    if (x >= 2.*beta) kappa = 1.;
  }

  return kappa;
}

void addElement(int const & id1, int const & id2, set<int> & nSet,
                unsigned int & nbElem, vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx,
                vector<vector<double>> & elemSubMat, double const & inpEps, double const & kappa,
                bool const & debug, fstream & dbg) {
  if (id1 < 0) return; // Not allowed, only id2 can be < 0 (boundary condition).

  if (debug) dbg << id1 << " " << id2 << endl;

  vector<double> elemMat;
  if (id1 >= 0 && id2 >= 0) {
    nSet.insert(id1); nSet.insert(id2);
    unsigned int ePtrLast = elemPtr[elemPtr.size() - 1]; elemPtr.push_back(ePtrLast + 2);
    elemIdx.push_back(id1); elemIdx.push_back(id2);
    elemMat.push_back( 1. + inpEps); elemMat.push_back(-1.);
    elemMat.push_back(-1.);          elemMat.push_back( 1. + inpEps);
  }
  else if (id1 >= 0 && id2 < 0) { // Dirichlet boundary condition.
    nSet.insert(id1);
    unsigned int ePtrLast = elemPtr[elemPtr.size() - 1]; elemPtr.push_back(ePtrLast + 1);
    elemIdx.push_back(id1);
    elemMat.push_back(1. + inpEps); // Ignore ghost point "out of the matrix" (id2 < 0) : count only contribution of id1 > 0.
  }
  transform(elemMat.begin(), elemMat.end(), elemMat.begin(), [kappa](double const & d){return d*kappa;}); // elemMat *= kappa.

  elemSubMat.push_back(elemMat);
  nbElem++;
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
      if (opt == "--debug") {debug = true; dbg.open("debug.inp", ios::out);}
      if (opt == "--verbose") {verbose = true; cout << "getInput arguments: " << args << endl;}
    }

    // Laplacian size and dimensions.

    int laplaceSize = 0;
    if (dim == 1) laplaceSize =                size*weakScaling;
    if (dim == 2) laplaceSize = sqrt(     size*size*weakScaling);
    if (dim == 3) laplaceSize = cbrt(size*size*size*weakScaling);
    if (debug) dbg << "# laplace size: " << laplaceSize << endl << endl;

    int dim1D = 0, dim2D = 0, dim3D = 0;
    if (dim == 1) {dim1D = laplaceSize; dim2D =           1; dim3D =           1;}
    if (dim == 2) {dim1D = laplaceSize; dim2D = laplaceSize; dim3D =           1;}
    if (dim == 3) {dim1D = laplaceSize; dim2D = laplaceSize; dim3D = laplaceSize;}

    // Initialize kappa parameters.

    double beta = 1.;
    double alpha = 0.;
    double xMax = (double) (laplaceSize - 1); // Last point coordinate.
    if (kappaInterp == "quad") {
      alpha = (kappaMax - beta)/(xMax*xMax);
    }
    else if (kappaInterp == "lin") {
      alpha = (kappaMax - beta)/xMax;
    }
    else if (kappaInterp == "minmax") {
      alpha = kappaMax;
      beta = xMax/3.;
    }

    // Laplacian: compute div(kappa*grad(u)) with kappa which may vary over space.

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
                double kappaX = computeKappa(kappaInterp, alpha, (double) d1, beta);
                double kappaY = computeKappa(kappaInterp, alpha, (double) d2, beta);
                double kappaZ = computeKappa(kappaInterp, alpha, (double) d3, beta);
                double kappa = kappaX*kappaY*kappaZ;
                if (addBC) addElement(centralIdx, -1, nSet, nbElem, elemPtr, elemIdx, elemSubMat, inpEps, kappa, debug, dbg);
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
                double kappaX = computeKappa(kappaInterp, alpha, (double) d1, beta);
                double kappaY = computeKappa(kappaInterp, alpha, (double) d2, beta);
                double kappaZ = computeKappa(kappaInterp, alpha, (double) d3, beta);
                double kappa = kappaX*kappaY*kappaZ;
                addElement(centralIdx, neighborIdx, nSet, nbElem, elemPtr, elemIdx, elemSubMat, inpEps, kappa, debug, dbg);
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
