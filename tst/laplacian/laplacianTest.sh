#!/bin/bash -eu

rm -f -- *.log

RUN="$(./laplacianRun.sh strong)"
echo "$RUN" # Print run output.

RUN="$(./laplacianRun.sh weak)"
echo "$RUN" # Print run output.

if [[ "$(find . -maxdepth 1 -name '*.log' | wc -l)" -ne "928" ]]; then
  echo "ERROR: bad nb log - $(find . -maxdepth 1 -name '*.log' | wc -l)"
  exit 1
fi

./laplacianPlot.sh > ./laplacianPlot.log 2>&1
@DIFF_EXECUTABLE@ ./laplacianPlot.ref ./laplacianPlot.log
