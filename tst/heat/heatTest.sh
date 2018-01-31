#!/bin/bash -eu

rm -f -- *.log

RUN="$(./heatRun.sh strong)"
echo "$RUN" # Print run output.

RUN="$(./heatRun.sh weak)"
echo "$RUN" # Print run output.

if [[ "$(find . -maxdepth 1 -name '*.log' | wc -l)" -ne "928" ]]; then
  echo "ERROR: bad nb log - $(find . -maxdepth 1 -name '*.log' | wc -l)"
  exit 1
fi
