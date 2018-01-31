#ifndef __laplacianServices__
#define __laplacianServices__

#include <string>
#include <vector>

using namespace std;

int initLaplacian(int const & laplaceSize, string const & kappaInterp, double const & kappaMax,
                  double & alpha, double & beta);

int getLaplacian(double const & inpEps, bool const & bc,
                 string const & kappaInterp, double const & alpha, double const & beta,
                 double const & x, double const & y, double const & z,
                 vector<double> & elemMat);

#endif
