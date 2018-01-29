#!/bin/bash -eu

rm -f -- *.log

RUN="$(./laplacianRun.sh strong)"
echo "$RUN" # Print run output.

RUN="$(./laplacianRun.sh weak)"
echo "$RUN" # Print run output.

./laplacianPlot.sh > ./laplacianPlot.log 2>&1
@DIFF_EXECUTABLE@ ./laplacianPlot.ref ./laplacianPlot.log
