#!/bin/bash

RUN="$(./graphRun.sh strong)"
echo "$RUN" # Print run output.

RUN="$(./graphRun.sh weak)"
echo "$RUN" # Print run output.

./graphPlot.sh > ./graphPlot.log 2>&1
@DIFF_EXECUTABLE@ ./graphPlot.ref ./graphPlot.log
