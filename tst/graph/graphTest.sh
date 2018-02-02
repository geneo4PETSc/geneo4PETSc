#!/bin/bash -eu

function errorMsg {
  if [[ "$?" -ne "0" ]]; then echo "ERROR: check *.log"; fi
}
trap 'errorMsg' EXIT

rm -f -- *.log

RUN="$(./graphRun.sh strong)"
echo "$RUN" # Print run output.

RUN="$(./graphRun.sh weak)"
echo "$RUN" # Print run output.

if [[ "$(find . -maxdepth 1 -name '*.log' | wc -l)" -ne "928" ]]; then
  echo "ERROR: bad nb log - $(find . -maxdepth 1 -name '*.log' | wc -l)"
  exit 1
fi

./graphPlot.sh > ./graphPlot.log 2>&1
@DIFF_EXECUTABLE@ ./graphPlot.ref ./graphPlot.log
