/*
 * Square blocks of a given size are created:
 *   - There is one central block.
 *   - Additional levels (concentric and connected blocks) can be added around the central block.
 *   - Each level is connected to the previous one.
 * Each border of each block can be connected to one node (named ground).
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm> // sort.
#include <set>
#include <vector>
#include <tuple>

using namespace std;

typedef tuple<vector<int>, vector<int>, vector<int>, vector<int>> border;

void addElement(unsigned int const & id1, unsigned int const & id2,
                set<unsigned int> & nSet,
                unsigned int & nbElem, vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx,
                double const & l, vector<vector<double>> & elemSubMat, double const & inpEps,
                bool const & debug, fstream & dbg) {
  if (debug) dbg << id1 << " " << id2 << endl;
  nSet.insert(id1); nSet.insert(id2);
  elemPtr.push_back(2*nbElem);
  elemIdx.push_back(id1); elemIdx.push_back(id2);
  vector<double> elemMat;
  elemMat.push_back(l*( 1.+inpEps)); elemMat.push_back(l* -1.);
  elemMat.push_back(l* -1.);         elemMat.push_back(l*( 1.+inpEps));
  elemSubMat.push_back(elemMat);
  nbElem++;
}

void buildBlock(string const & msg, int const & blockSize, int & nodeID, vector<border> & borders, bool central,
                set<unsigned int> & nSet,
                unsigned int & nbElem, vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx,
                double const & l, vector<vector<double>> & elemSubMat, double const & inpEps, bool const & noGround,
                bool const & debug, fstream & dbg) {
  // Build a block.

  if (debug) dbg << "# " << msg << endl;
  for (int i = 0; i < blockSize; i++) { // Rows of squared block.
    for (int j = 0; j < blockSize-1; j++) {
      unsigned int const id1 = nodeID+j; unsigned int const id2 = nodeID+(j+1);
      addElement(id1, id2, nSet, nbElem, elemPtr, elemIdx, l, elemSubMat, inpEps, debug, dbg);
    }
    nodeID += blockSize;
  }
  int nID = nodeID-1;
  for (int i = 0; i < blockSize; i++) { // Columns of squared blocks.
    for (int j = 0; j < blockSize-1; j++) {
      unsigned int const id1 = nID-(j*blockSize); unsigned int const id2 = nID-((j+1)*blockSize);
      addElement(id1, id2, nSet, nbElem, elemPtr, elemIdx, l, elemSubMat, inpEps, debug, dbg);
    }
    nID -= 1;
  }
  if (debug) dbg << endl;

  // Build block borders: up, right, down, left.

  nID = nodeID-1;
  vector<int> up, right, down, left;
  for (int i = 0; i < blockSize; i++) down.push_back(nID-i);
  for (int i = 0; i < blockSize; i++) right.push_back(nID-i*blockSize);
  for (int i = 0; i < blockSize; i++)  left.push_back(nID-i*blockSize-(blockSize-1));
  for (int i = 0; i < blockSize; i++) up.push_back(nID-(blockSize-1)*blockSize-i);
  sort(  up.begin(),   up.end()); sort(right.begin(), right.end());
  sort(down.begin(), down.end()); sort( left.begin(),  left.end());

  // Export block borders to create later connections between blocks.

  borders.push_back(border(up, right, down, left));
  if (central) { // Fake central level is a level with 4 identical blocks.
    borders.push_back(border(up, right, down, left));
    borders.push_back(border(up, right, down, left));
    borders.push_back(border(up, right, down, left));
  }

  if (noGround) return;

  // Connect block to the ground (0).

  if (debug) dbg << "# Up border connected to the ground - " << msg << endl;
  for (int i = 0; i < blockSize; i++) {
    unsigned int const id1 = up[i]; unsigned int const id2 = 0;
    addElement(id1, id2, nSet, nbElem, elemPtr, elemIdx, l, elemSubMat, inpEps, debug, dbg);
  }
  if (debug) dbg << endl;

  if (debug) dbg << "# Right border connected to the ground - " << msg << endl;
  for (int i = 0; i < blockSize; i++) {
    unsigned int const id1 = right[i]; unsigned int const id2 = 0;
    addElement(id1, id2, nSet, nbElem, elemPtr, elemIdx, l, elemSubMat, inpEps, debug, dbg);
  }
  if (debug) dbg << endl;

  if (debug) dbg << "# Down border connected to the ground - " << msg << endl;
  for (int i = 0; i < blockSize; i++) {
    unsigned int const id1 = down[i]; unsigned int const id2 = 0;
    addElement(id1, id2, nSet, nbElem, elemPtr, elemIdx, l, elemSubMat, inpEps, debug, dbg);
  }
  if (debug) dbg << endl;

  if (debug) dbg << "# Left border connected to the ground - " << msg << endl;
  for (int i = 0; i < blockSize; i++) {
    unsigned int const id1 = left[i]; unsigned int const id2 = 0;
    addElement(id1, id2, nSet, nbElem, elemPtr, elemIdx, l, elemSubMat, inpEps, debug, dbg);
  }
  if (debug) dbg << endl;
}

extern "C" {
  int getInput(string const & args,
               unsigned int & nbElem, unsigned int & nbNode,
               vector<unsigned int> & elemPtr, vector<unsigned int> & elemIdx, vector<vector<double>> & elemSubMat) {
    // Check arguments.

    int size = 4, level = 1, weakScaling = 1; double inpEps = 0.0001; bool noGround = false;
    bool verbose = false, debug = false; fstream dbg;

    if (debug) dbg << "# args: " << args << endl << endl;
    stringstream ssArgs(args);
    while (ssArgs) {
      string opt; ssArgs >> opt; // Command line option
      if (opt == "--size") {
        ssArgs >> size;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--level") {
        ssArgs >> level;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--weakScaling") {
        ssArgs >> weakScaling;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--inpEps") {
        ssArgs >> inpEps;
        if (!ssArgs) {cerr << "Error: invalid command line" << endl; return 1;}
      }
      if (opt == "--noGround") noGround = true; // Suppress ground (problem defined up to a constant): make G invertible.
      if (opt == "--debug") {debug = true; dbg.open("debug.inp", ios::out);}
      if (opt == "--verbose") {verbose = true; cout << "getInput arguments: " << args << endl;}
    }

    // Central block.

    int blockSize = (int) sqrt(size*weakScaling);
    if (debug) dbg << "# block size: " << blockSize << endl << endl;

    int nodeID = noGround ? 0 : 1;
    set<unsigned int> nSet;
    vector<border> borders; // up, right, down, left.
    buildBlock("central block", blockSize, nodeID, borders, true,
               nSet, nbElem, elemPtr, elemIdx, 1., elemSubMat, inpEps, noGround,
               debug, dbg);

    // Levels.

    for (int l = 1; l <= level; l++) {
      for (int b = 0; b < 4; b++) { // Each level is made of 4 blocks (up, right, down, left).
        stringstream ss; ss << "level " << l << ": block " << b;
        buildBlock(ss.str(), blockSize, nodeID, borders, false,
                   nSet, nbElem, elemPtr, elemIdx, l+1., elemSubMat, inpEps, noGround,
                   debug, dbg);
      }

      for (int b = 0; b < 4; b++) { // Each level is made of 4 blocks (up, right, down, left).
        if (debug) dbg << "# Connect horizontally - level " << l << endl;
        int bBefore = b; int bAfter = ((b+1) < 4) ? b+1 : 0;
        vector<int> from, to;
        if      (b == 0) {from = get<1>(borders[4*l+bBefore]); to = get<0>(borders[4*l+bAfter]);}
        else if (b == 1) {from = get<2>(borders[4*l+bBefore]); to = get<1>(borders[4*l+bAfter]);}
        else if (b == 2) {from = get<3>(borders[4*l+bBefore]); to = get<2>(borders[4*l+bAfter]);}
        else             {from = get<0>(borders[4*l+bBefore]); to = get<3>(borders[4*l+bAfter]);}
        for (unsigned int i = 0; i < from.size(); i++) {
          unsigned int const id1 = from[i]; unsigned int const id2 = to[i];
          addElement(id1, id2, nSet, nbElem, elemPtr, elemIdx, 0.5*(l+1.), elemSubMat, inpEps, debug, dbg);
        }
        if (debug) dbg << endl;
      }

      for (int b = 0; b < 4; b++) { // Each level is made of 4 blocks (up, right, down, left).
        if (debug) dbg << "# Connect vertically - level " << l << endl;
        vector<int> from, to;
        if      (b == 0) {from = get<0>(borders[4*(l-1)+b]); to = get<2>(borders[4*l+b]);}
        else if (b == 1) {from = get<1>(borders[4*(l-1)+b]); to = get<3>(borders[4*l+b]);}
        else if (b == 2) {from = get<2>(borders[4*(l-1)+b]); to = get<0>(borders[4*l+b]);}
        else             {from = get<3>(borders[4*(l-1)+b]); to = get<1>(borders[4*l+b]);}
        for (unsigned int i = 0; i < from.size(); i++) {
          unsigned int const id1 = from[i]; unsigned int const id2 = to[i];
          addElement(id1, id2, nSet, nbElem, elemPtr, elemIdx, 0.5*(l+1.), elemSubMat, inpEps, debug, dbg);
        }
        if (debug) dbg << endl;
      }
    }
    elemPtr.push_back(2*nbElem);
    nbNode = nSet.size();

    if (verbose) cout << "getInput: nbNode " << nbNode << ", nbElem " << nbElem << endl;

    return 0;
  }
}
