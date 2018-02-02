#!/bin/bash -eu

function errorMsg {
  if [[ "$?" -ne "0" ]]; then echo "ERROR: check *.log"; fi
}
trap 'errorMsg' EXIT

rm -f -- *.log

RUN="$(./heatRun.sh strong)"
echo "$RUN" # Print run output.

RUN="$(./heatRun.sh weak)"
echo "$RUN" # Print run output.

if [[ "$(find . -maxdepth 1 -name '*.log' | wc -l)" -ne "928" ]]; then
  echo "ERROR: bad nb log - $(find . -maxdepth 1 -name '*.log' | wc -l)"
  exit 1
fi
