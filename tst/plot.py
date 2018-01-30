#!/usr/bin/env python
# coding: utf-8
"""This script is designed to plot geneo4PETSc results"""

# Import modules

from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import glob
import sys
from itertools import cycle
import math
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy

# Classes.

class job(object):
    """A job is a run with a set of given options and associated outcomes"""
    def __init__(self):
        """Initialization"""
        self.fn = "" # File name.
        self.ws = 1 # Weak scaling (1 means strong scaling - n > 1 means weak scaling).
        self.metis = "" # Metis mode.
        self.overlap = "0" # Overlap layer.
        self.nbDOF = 0 # Number of DOF.
        self.nbCoef = 0 # Number of coefficients.
        self.estimDimE = -1 # Number of estimated eigen values.
        self.estimDimEMin = -1 # Minimum number of estimated eigen values.
        self.estimDimEMax = -1 # Maximum number of estimated eigen values.
        self.realDimE = -1 # Number of real eigen values.
        self.realDimEMin = -1 # Minimum number of real eigen values.
        self.realDimEMax = -1 # Maximum number of real eigen values.
        self.nicolaides = -1 # Maximum number of nicolaides vectors.
        self.ksp = "" # KSP.
        self.pc = None # Preconditioner.
        self.offload = False # Offload.
        self.L1 = None # Level 1.
        self.tau = None # Tau (GenEO).
        self.gamma = None # Gamma (GenEO).
        self.L2 = None # Level 2.
        self.optim = None # Optim parameter (Robin).
        self.nbIt = 0 # Number of iterations.
        self.readInp = 0. # Time for reading input.
        self.partDecomp = 0. # Time for partitioning and decomposing.
        self.createA = 0. # Time for creating A.
        self.setUpSolve = 0. # Time for setting up the KSP solver.
        self.itSolve = 0. # Time for KSP solver iterations.
        self.solve = 0. # Time for solve (set up + iterations).

    def buildJob(self, fn, lines):
        """Build a job from its log file"""
        self.fn = fn
        for token in fn.split("-"):
            if token.find("ws=") != -1:
                self.ws = int(token.split("=")[1])
        if len(lines) <= 6:
            sys.exit("Error: can not read file " + fn)
        line0 = lines[0].split()
        for idx, token in enumerate(line0):
            if token == "DOFs":
                self.nbDOF = int(line0[idx+1].replace(",", ""))
            if token == "coefs":
                self.nbCoef = int(line0[idx+1].replace(",", ""))
            if token == "metis":
                self.metis = line0[idx+1].replace(",", "")
            if token == "overlap":
                self.overlap = line0[idx+1].replace(",", "")
        line1 = lines[1].split()
        for idx, token in enumerate(line1):
            if token.find("ksp") != -1:
                self.ksp = line1[idx-1].replace(",", "")
        line2 = lines[2].split()
        for idx, token in enumerate(line2):
            if token.find("pc") != -1:
                self.pc = line2[idx-1].replace(",", "")
            if token.find("offload") != -1:
                self.offload = True
            if token == "L1":
                self.L1 = line2[idx+1].replace(",", "")
            if token == "tau":
                self.tau = line2[idx+1].replace(",", "")
            if token == "gamma":
                self.gamma = line2[idx+1].replace(",", "")
            if token == "optim":
                self.optim = line2[idx+1].replace(",", "")
            if token == "L2":
                s1 = line2[idx+1].replace(",", "")
                s2 = line2[idx+2].replace(",", "")
                self.L2 = s1 + "+" + s2
        line3 = lines[3].split()
        for idx, token in enumerate(line3):
            if token == "estim":
                self.estimDimE = int(line3[idx+2])
                self.estimDimEMin = int(line3[idx+5].replace(",", ""))
                self.estimDimEMax = int(line3[idx+7].replace("),", ""))
            if token == "real":
                self.realDimE = int(line3[idx+2])
                self.realDimEMin = int(line3[idx+5].replace(",", ""))
                self.realDimEMax = int(line3[idx+7].replace("),", ""))
            if token == "nicolaides":
                self.nicolaides = int(line3[idx+1])
        line4 = lines[4].split()
        self.nbIt = int(line4[5].replace(",", ""))
        line5 = lines[5].split()
        self.readInp = float(line5[3].replace(",", ""))
        self.partDecomp = float(line5[8].replace(",", ""))
        self.createA = float(line5[12].replace(",", ""))
        self.setUpSolve = float(line5[17].replace(",", ""))
        self.itSolve = float(line5[21].replace(",", ""))
        self.solve = float(line5[24].replace(",", ""))

    def getSurfName(self):
        """Get surface name the job belongs to"""
        surfName = "metis=" + self.metis + "-" + "overlap=" + self.overlap
        surfName += "-" + "ksp=" + self.ksp + "-" + "pc=" + self.pc
        if self.pc.find("geneo") != -1:
            fields = ["L1", "tau", "gamma", "L2", "optim"]
            attrs = [self.L1, self.tau, self.gamma, self.L2, self.optim]
            for field, attr in zip(fields, attrs):
                if attr is not None:
                    surfName += "-" + field + "=" + attr
        if self.pc.find("geneo") != -1:
            surfName += ("-" + "offloadE") if self.offload else ("-" + "distribE")
        return surfName

class surf(object):
    """A surface is a list of jobs"""
    def __init__(self):
        """Initialization"""
        self.surfName = None
        self.x = None
        self.y = None
        self.z = None

    def addJob(self, j, args, tIdx, nIdx):
        """Add job to surface"""
        # Init surface if not created.
        if not self.surfName:
            self.surfName = j.getSurfName()
            self.z = {}
            negInt = numpy.array([-1]*(len(args.np)*len(args.tol))).reshape(self.x.shape)
            self.z["nbIt"] = numpy.copy(negInt)
            self.z["nbDOF"] = numpy.copy(negInt)
            self.z["nbCoef"] = numpy.copy(negInt)
            self.z["estimDimE"] = numpy.copy(negInt)
            self.z["estimDimEMin"] = numpy.copy(negInt)
            self.z["estimDimEMax"] = numpy.copy(negInt)
            self.z["realDimE"] = numpy.copy(negInt)
            self.z["realDimEMin"] = numpy.copy(negInt)
            self.z["realDimEMax"] = numpy.copy(negInt)
            self.z["nicolaides"] = numpy.copy(negInt)
            negDouble = numpy.array([-1]*(len(args.np)*len(args.tol))).reshape(self.x.shape)
            self.z["readInp"] = numpy.copy(negDouble)
            self.z["partDecomp"] = numpy.copy(negDouble)
            self.z["createA"] = numpy.copy(negDouble)
            self.z["setUpSolve"] = numpy.copy(negDouble)
            self.z["itSolve"] = numpy.copy(negDouble)
            self.z["solve"] = numpy.copy(negDouble)
        # Check.
        if self.surfName != j.getSurfName():
            return False
        # Fill surface.
        self.z["nbIt"][tIdx, nIdx] = j.nbIt
        self.z["nbDOF"][tIdx, nIdx] = j.nbDOF
        self.z["nbCoef"][tIdx, nIdx] = j.nbCoef
        self.z["estimDimE"][tIdx, nIdx] = j.estimDimE
        self.z["estimDimEMin"][tIdx, nIdx] = j.estimDimEMin
        self.z["estimDimEMax"][tIdx, nIdx] = j.estimDimEMax
        self.z["realDimE"][tIdx, nIdx] = j.realDimE
        self.z["realDimEMin"][tIdx, nIdx] = j.realDimEMin
        self.z["realDimEMax"][tIdx, nIdx] = j.realDimEMax
        self.z["nicolaides"][tIdx, nIdx] = j.nicolaides
        self.z["readInp"][tIdx, nIdx] = j.readInp
        self.z["partDecomp"][tIdx, nIdx] = j.partDecomp
        self.z["createA"][tIdx, nIdx] = j.createA
        self.z["setUpSolve"][tIdx, nIdx] = j.setUpSolve
        self.z["itSolve"][tIdx, nIdx] = j.itSolve
        self.z["solve"][tIdx, nIdx] = j.solve
        return True

    def computeSpeedup(self, args):
        """Compute speedup"""
        # Compute speed up.
        surfSolve = self.z["solve"]
        negDouble = numpy.array([-1.]*(len(args.np)*len(args.tol))).reshape(surfSolve.shape)
        self.z["speedup"] = negDouble
        for tIdx in range(len(args.tol)):
            refTime = surfSolve[tIdx, 0]
            for nIdx in range(len(args.np)):
                jobTime = surfSolve[tIdx, nIdx]
                if math.fabs(jobTime) > 1.e-12:
                    self.z["speedup"][tIdx, nIdx] = refTime/jobTime

# Functions

def getJobs(fpattern, n, t, pc, jobs, debug):
    """Get all job outcomes from their log files"""
    if fpattern is None:
        return
    fs = "*" + fpattern + "*" # File search.
    if not glob.glob(fs + ".log"):
        sys.exit("Error: no file named " + fs + ".log")
    fs = fs + "np=" + n + "*"
    fs = fs + "tol=" + t + "*"
    fs = fs + "pc=" + pc + "*"
    fs = fs + ".log"
    for fn in glob.glob(fs): # File name.
        lines = open(fn).readlines()
        lines = [line for line in lines if line[0:3] != "WRNG"] # Suppress warnings.
        lines = [line for line in lines if len(line.split()) > 0] # Suppress empty lines.
        if len(lines) >= 4:
            line4 = lines[4].split()
            if line4[3].replace(",", "") != "converged":
                print("Error: " + fn + " has not converged")
                continue # In case of partial runs (not all jobs have passed)
        j = job()
        j.buildJob(fn, lines)
        if j.ws not in jobs[n][t]:
            jobs[n][t][j.ws] = [] # Several jobs for the same n, t, ws (with different ksp, pc, ...)
        jobs[n][t][j.ws].append(j)
        if debug:
            print("Debug: ", fn)
            print(vars(j))
            print("")

def skipSurf(surfName, args):
    """Skip surface according to filters"""
    isInInc = True
    for fi in args.filterInc:
        fiFoundInAttr = False
        for attr in surfName.split("-"):
            if attr == fi:
                fiFoundInAttr = True
                break
        isInInc = isInInc and fiFoundInAttr # Include if all keyword are found.
    isInExc = False
    for fj in args.filterExc:
        for attr in surfName.split("-"):
            if attr == fj:
                isInExc = True # Exclude if one keyword is found.
                break
    skip = not isInInc or isInExc
    if args.debug and skip:
        print("Debug: skip surf ", isInInc, isInExc, surfName)
    return skip

def getLabelFromSurfName(surfName, args):
    """Get label from surface name"""
    label = surfName.split("-")
    for l in args.label2Title:
        try:
            label.remove(l)
        except RuntimeError:
            args.label2Title.remove(l) # Nothing to remove, nothing to move to title !
    return "-".join(label)

def getAxisLabel(axis):
    """Get axis label"""
    if axis == "nbIt":
        return "nb iterations"
    if axis == "nbDOF":
        return "nb DOF"
    if axis == "nbCoef":
        return "nb coefficients"
    if axis == "estimDimE":
        return "estimated dim E"
    if axis == "estimDimEMin":
        return "min estimated local dim E"
    if axis == "estimDimEMax":
        return "max estimated local dim E"
    if axis == "realDimE":
        return "dim E"
    if axis == "realDimEMin":
        return "min local dim E"
    if axis == "realDimEMax":
        return "max local dim E"
    if axis == "nicolaides":
        return "number of nicolaides vectors"
    if axis == "readInp":
        return "read input time (s)"
    if axis == "partDecomp":
        return "partition / decompose time (s)"
    if axis == "createA":
        return "create A time (s)"
    if axis == "setUpSolve":
        return "solve setup time (s)"
    if axis == "itSolve":
        return "solve iterations time (s)"
    if axis == "solve":
        return "solve time (s)"
    if axis == "speedup":
        return "speedup"
    sys.exit("Error: unknown axis")

def plotLabels(args, axis1, axis2, axis3, scaling):
    """Plot labels"""
    # Label axis.
    if args.plot2D:
        axis1.set_xlabel("nb partitions")
        axis1.set_ylabel(getAxisLabel(args.axis1))
        axis2.set_xlabel("nb partitions")
        axis2.set_ylabel(getAxisLabel(args.axis2))
        axis3.set_xlabel("nb partitions")
        axis3.set_ylabel(getAxisLabel(args.axis3))
        for axis in [axis1, axis2, axis3]:
            axis.set_xticks([int(n) for n in args.np])
            axis.set_xticklabels([n for n in args.np])
            axis.set_xscale("log", basex=2)
            axis.legend(loc="best")
    else:
        axis1.set_xlabel("nb partitions")
        axis1.set_ylabel("tolerance")
        axis1.set_zlabel(getAxisLabel(args.axis1))
        axis2.set_xlabel("nb partitions")
        axis2.set_ylabel("tolerance")
        axis2.set_zlabel(getAxisLabel(args.axis2))
        axis3.set_xlabel("nb partitions")
        axis3.set_ylabel("tolerance")
        axis3.set_zlabel(getAxisLabel(args.axis3))
        for axis in [axis1, axis2, axis3]:
            axis.set_xticks([int(n) for n in args.np])
            axis.set_xticklabels([n for n in args.np])
            axis.set_yticks([float(t) for t in args.tol])
            axis.set_yticklabels(["%1.e"%float(t) for t in args.tol])
    axis1.set_title(("strong scaling, " if scaling == "strong" else "weak scaling, ") + getAxisLabel(args.axis1))
    axis2.set_title(("strong scaling, " if scaling == "strong" else "weak scaling, ") + getAxisLabel(args.axis2))
    axis3.set_title(("strong scaling, " if scaling == "strong" else "weak scaling, ") + getAxisLabel(args.axis3))

def plotSurfs(args, fig, scaling, surfs):
    """Plot surfaces"""
    # Create axis.
    rowPlotID, nbRowPlot = 0, 1 # 1st line of plots.
    if args.strong and args.weak:
        nbRowPlot = 2
        if scaling == "weak":
            rowPlotID = 1 # Add a 2d line of plots
    axis1, axis2, axis3 = None, None, None
    if args.plot2D:
        axis1 = fig.add_subplot(nbRowPlot, 3, 1 + rowPlotID * 3)
        axis2 = fig.add_subplot(nbRowPlot, 3, 2 + rowPlotID * 3)
        axis3 = fig.add_subplot(nbRowPlot, 3, 3 + rowPlotID * 3)
    else:
        axis1 = fig.add_subplot(nbRowPlot, 3, 1 + rowPlotID * 3, projection="3d")
        axis2 = fig.add_subplot(nbRowPlot, 3, 2 + rowPlotID * 3, projection="3d")
        axis3 = fig.add_subplot(nbRowPlot, 3, 3 + rowPlotID * 3, projection="3d")

    # Plot surfaces.
    legendLines = []
    legendTitles = []
    color = cycle(plt.cm.get_cmap("rainbow")(numpy.linspace(0, 1, len(surfs[scaling].keys()))))
    for surfName in sorted(surfs[scaling].keys()):
        print("Info:   => plotting surface " + surfName + " (" + scaling + " scaling)")
        c = next(color)
        s = surfs[scaling][surfName]
        if args.plot2D:
            lStyle = cycle(["-", "--", "-.", ":"])
            for tIdx, t in enumerate(args.tol):
                l = "tol = " + t
                ls = next(lStyle)
                axis1.plot(s.x[tIdx, :], s.z[args.axis1][tIdx, :], color=c, label=l, linestyle=ls)
                axis2.plot(s.x[tIdx, :], s.z[args.axis2][tIdx, :], color=c, label=l, linestyle=ls)
                axis3.plot(s.x[tIdx, :], s.z[args.axis3][tIdx, :], color=c, label=l, linestyle=ls)
        else:
            axis1.plot_surface(s.x, s.y, s.z[args.axis1], color=c)
            axis2.plot_surface(s.x, s.y, s.z[args.axis2], color=c)
            axis3.plot_surface(s.x, s.y, s.z[args.axis3], color=c)
        legendLines.append(plt.Line2D([0], [0], linestyle="none", color=c, marker="o"))
        legendTitles.append(getLabelFromSurfName(surfName, args))
        if args.debug:
            print("Debug: plot ", scaling, " scaling for ", surfName)
    plotLabels(args, axis1, axis2, axis3, scaling)
    return legendLines, legendTitles

def plotScaling(args, scaling, jobs, fig):
    """Plot jobs according to strong and weak scaling"""
    # Skip plot if not necessary.
    if scaling == "strong":
        if not args.strong:
            return [], [], ""
    if scaling == "weak":
        if not args.weak:
            return [], [], ""

    # Get job surfaces (to be plotted).
    x, y = numpy.meshgrid([int(n) for n in args.np], [float(t) for t in args.tol])
    surfs = {}
    if scaling not in surfs:
        surfs[scaling] = {}
    nbDOF, nbCoef, nbDOFPerProc, nbValPerProc, nbJob = 0, 0, 0, 0, 0
    for nIdx, n in enumerate(args.np):
        if n not in jobs:
            continue # In case of partial runs (not all jobs have passed)
        for tIdx, t in enumerate(args.tol):
            if t not in jobs[n]:
                continue # In case of partial runs (not all jobs have passed)
            ws = int(args.np[0]) if scaling == "strong" else int(n)
            if ws not in jobs[n][t]:
                continue # In case of partial runs (not all jobs have passed)
            for j in jobs[n][t][ws]:
                # Filter.
                if scaling == "strong" and j.fn.find(args.strong) == -1:
                    continue # Not eligible for strong scaling.
                if scaling == "weak" and j.fn.find(args.weak) == -1:
                    continue # Not eligible for weak scaling.
                if skipSurf(j.getSurfName(), args):
                    continue # Filter.
                # Retrieve surface if already created.
                s = None
                if j.getSurfName() in surfs[scaling]:
                    s = surfs[scaling][j.getSurfName()]
                else:
                    s = surf()
                    s.x, s.y = numpy.copy(x), numpy.copy(y)
                # Add job into surface.
                added = s.addJob(j, args, tIdx, nIdx)
                # Update informations to build title (later).
                if added:
                    nbDOF = nbDOF + j.nbDOF
                    nbCoef = nbCoef + j.nbCoef
                    nbDOFPerProc = nbDOFPerProc + j.nbDOF/int(n)
                    nbValPerProc = nbValPerProc + j.nbCoef/int(n)
                    nbJob = nbJob + 1
                # Add surface if not already stored.
                if s.surfName not in surfs[scaling]:
                    surfs[scaling][s.surfName] = s
    for surfName in surfs[scaling]:
        surfs[scaling][surfName].computeSpeedup(args)
    if not surfs[scaling].keys():
        sys.exit("Error: no surface to plot")
    print("Info: " + str(len(surfs[scaling].keys())) + " surfaces to plot (" + scaling + " scaling)")

    # Build title.
    supTitle = scaling + " scaling"
    if scaling == "strong":
        supTitle = supTitle + " (" + str(nbDOF/nbJob) + " DOFs, " + str(nbCoef/nbJob) + " coefs)"
    if scaling == "weak":
        supTitle = supTitle + " (" + str(nbDOFPerProc/nbJob) + " DOF/proc, " + str(nbValPerProc/nbJob) + " values/proc)"

    # Plot surfaces.
    lines, titles = plotSurfs(args, fig, scaling, surfs)
    return lines, titles, supTitle

def getDefaultParameters(fpattern, key, params, sortReverse):
    """Get default parameters from file names"""
    if params:
        return # No need to find default values: the user has provided them.

    # Look for logs that match the base pattern.
    d = {}
    for fp in fpattern: # File pattern.
        if fp is None:
            continue
        for fn in glob.glob("*" + fp + "*"): # File name.
            tokens = re.split("(?:-)(?=[a-z]+)", fn) # Split according to "-" if "-" is followed by a string [a-z].
            for token in tokens: # Handle "np=01" and "tol=1.e-06" that are connected by "-".
                if token.find(key) != -1:
                    v = token.split("=")[1]
                    v = v.replace(".log", "") # Suppress .log if token is at the end of the file name.
                    k = None
                    try:
                        k = float(v)
                    except ValueError:
                        k = v
                    d[k] = v

    # List keys that match the base pattern.
    for k in sorted(d.keys(), reverse=sortReverse):
        params.append(d[k])

def getArgs():
    """Get program arguments"""
    # Build parser to handle args.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Visualize geneo4PETSc results.')
    parser.add_argument("-sg", "--strong", nargs="?", default=None, type=str, help="base name of strong scaling files (size=*-level=*)")
    parser.add_argument("-wk", "--weak", nargs="?", default=None, type=str, help="base name of weak scaling files (size=*-level=*)")
    parser.add_argument("-n", "--np", nargs="*", default=[], help="list number of processus")
    parser.add_argument("-t", "--tol", nargs="*", default=[], help="tolerance")
    parser.add_argument("-pc", "--pc", nargs="*", default=[], help="preconditioner")
    axisHelp = "data plotted on %s (nbDOF, nbCoef, estimDimE, estimDimEMin, estimDimEMax, realDimE, realDimEMin, realDimEMax, nicolaides, "
    axisHelp += "readInp, partDecomp, createA, setUpSolve, itSolve, solve, speedup, nbIt)"
    parser.add_argument("-a1", "--axis1", nargs="?", default="solve", help=(axisHelp % "axis1"))
    parser.add_argument("-a2", "--axis2", nargs="?", default="speedup", help=(axisHelp % "axis2"))
    parser.add_argument("-a3", "--axis3", nargs="?", default="nbIt", help=(axisHelp % "axis3"))
    parser.add_argument("-fi", "--filterInc", nargs="*", default=[], help="include surface filter whose name matches all pattern(s)")
    parser.add_argument("-fe", "--filterExc", nargs="*", default=[], help="exclude surface filter among what is included")
    parser.add_argument("-l2t", "--label2Title", nargs="*", default=[], help="move token(s) from labels to title")
    parser.add_argument("-2D", "--plot2D", action="store_true", help="plot 2D graphs")
    parser.add_argument("-d", "--debug", action="store_true", help="add debug print")
    parser.add_argument("-bs", "--bspace", type=float, default=0.1, help="white space below subplots")
    parser.add_argument("-ts", "--tspace", type=float, default=0.9, help="white space above subplots")
    parser.add_argument("-ls", "--lspace", type=float, default=0.1, help="white space at the left of subplots")
    parser.add_argument("-rs", "--rspace", type=float, default=0.9, help="white space at the right of subplots")
    parser.add_argument("-hs", "--hspace", type=float, default=0.2, help="white space between subplots (height)")
    parser.add_argument("-ws", "--wspace", type=float, default=0.2, help="white space between subplots (width)")
    parser.add_argument("-nlc", "--nblegcol", type=int, default=1, help="number of columns in the legend")
    parser.add_argument("-sv", "--save", default=False, type=str, help="save plot as a figure")
    args = parser.parse_args()
    if not args.strong and not args.weak:
        sys.exit("Error: need a base name at least for strong or weak scaling")
    return args

# Main program

def main():
    """Main function of the module"""
    args = getArgs()
    getDefaultParameters([args.strong, args.weak], "np=", args.np, False)
    getDefaultParameters([args.strong, args.weak], "tol=", args.tol, True)
    getDefaultParameters([args.strong, args.weak], "pc=", args.pc, False)
    if args.strong:
        fs = "*" + args.strong + "*" + ".log"
        print("Info: " + str(len(glob.glob(fs))) + " file(s) named " + fs)
    if args.weak:
        fs = "*" + args.weak + "*" + ".log"
        print("Info: " + str(len(glob.glob(fs))) + " file(s) named " + fs)

    # Read logs.
    jobs = {}
    for n in args.np:
        if n not in jobs:
            jobs[n] = {}
        for t in args.tol:
            if t not in jobs[n]:
                jobs[n][t] = {}
            for pc in args.pc:
                getJobs(args.strong, n, t, pc, jobs, args.debug)
                getJobs(args.weak, n, t, pc, jobs, args.debug)

    # Plot.
    fig = plt.figure()
    plt.rcParams["savefig.directory"] = os.getcwd() # Default directory to save figure is the current directory.
    strongLines, strongTitles, strongSupTitle = plotScaling(args, "strong", jobs, fig)
    weakLines, weakTitles, weakSupTitle = plotScaling(args, "weak", jobs, fig)

    # Build legend and title.
    linesAndTitles = {}
    for t, l in zip(strongTitles + weakTitles, strongLines + weakLines):
        linesAndTitles[t] = (l, t) # Remove duplicates using dictionary.
    lines = [linesAndTitles[t][0] for t in sorted(linesAndTitles.keys())]
    titles = [linesAndTitles[t][1] for t in sorted(linesAndTitles.keys())]
    if args.strong and args.weak:
        fig.legend(lines, titles, loc="center", bbox_to_anchor=(0.51, 0.49), bbox_transform=fig.transFigure, ncol=args.nblegcol)
    else:
        fig.legend(lines, titles, loc="lower center", bbox_to_anchor=(0.51, 0.05), bbox_transform=fig.transFigure, ncol=args.nblegcol)
    supTitle = strongSupTitle if strongSupTitle else ""
    if args.strong and args.weak:
        supTitle += " and "
    supTitle += weakSupTitle if weakSupTitle else ""
    supTitle += "" if not args.label2Title else (" with " + ", ".join(args.label2Title))
    plt.suptitle(supTitle)

    # Show plot.
    plt.subplots_adjust(bottom=args.bspace, top=args.tspace, left=args.lspace, right=args.rspace, hspace=args.hspace, wspace=args.wspace)
    if args.save:
        plt.savefig(args.save, dpi=fig.dpi)
    else:
        plt.show()

if __name__ == "__main__":
    main()
