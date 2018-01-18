#!/bin/bash

../plot.py -sv plot1.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres"
../plot.py -sv plot2.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres" "pc=bjacobi"
