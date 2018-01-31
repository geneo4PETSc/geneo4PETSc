#include "laplacianServices.hpp"

#include <algorithm> // transform.

using namespace std;

int initLaplacian(int const & laplaceSize, string const & kappaInterp, double const & kappaMax,
                  double & alpha, double & beta) {
  alpha = 0.; beta = 1.;

  // Initialize kappa parameters.

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

  return 0;
}

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

int getLaplacian(double const & inpEps, bool const & bc,
                 string const & kappaInterp, double const & alpha, double const & beta,
                 double const & x, double const & y, double const & z,
                 vector<double> & elemMat) {
  elemMat.clear();

  /*              1                              1            */
  /*              .                             .--.          */
  /*             / \                            |  |          */
  /*            /   \ 0                       0 |  |          */
  /*  phi_i  --o  o  o--o--  =>  grad(phi_i)  --o  o  o--o--  */
  /*              i  j                            i|  |j      */
  /*                                               |  |       */
  /*                                               .--.       */
  /*                                                -1.       */

  /*                 1                              1         */
  /*                 .                             .--.       */
  /*                / \                            |  |       */
  /*               /   \ 0                        i|  |j      */
  /*  phi_j  --o--o  o  o--  =>  grad(phi_j)  --o--o  o  o--  */
  /*              i  j                                |  | 0  */
  /*                                                  |  |    */
  /*                                                  .--.    */
  /*                                                   -1.    */

  //                  i     j
  //              | l_ii  l_ij | i
  //  laplacian = |            |
  //              | l_ji  l_jj | j
  //
  //  distance(i, j) = d = 1.
  //
  //  l_ii = int_[i,j](grad(phi_i).grad(phi_i)) = d*(-1.)*(-1.) =  1.
  //  l_ij = int_[i,j](grad(phi_i).grad(phi_j)) = d*(-1.)*( 1.) = -1.
  //  l_ji = int_[i,j](grad(phi_j).grad(phi_i)) = d*( 1.)*(-1.) = -1.
  //  l_jj = int_[i,j](grad(phi_j).grad(phi_j)) = d*( 1.)*( 1.) =  1.

  if (!bc) { // No boundary condition.
    elemMat.push_back( 1. + inpEps); elemMat.push_back(-1.);
    elemMat.push_back(-1.);          elemMat.push_back( 1. + inpEps);
  }
  else { // Dirichlet boundary condition.
    elemMat.push_back(1. + inpEps); // Ignore ghost point (1st DOF) "out of the matrix" : account only for the contribution of the 2d DOF.
  }

  double kappaX = computeKappa(kappaInterp, alpha, x, beta);
  double kappaY = computeKappa(kappaInterp, alpha, y, beta);
  double kappaZ = computeKappa(kappaInterp, alpha, z, beta);
  double kappa = kappaX*kappaY*kappaZ;
  transform(elemMat.begin(), elemMat.end(), elemMat.begin(), [kappa](double const & d){return d*kappa;}); // elemMat *= kappa.

  return 0;
}
