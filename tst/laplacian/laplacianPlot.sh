#!/bin/bash -eu

function errorMsg {
  if [[ "$?" -ne "0" ]]; then echo "ERROR: check *.log"; fi
}
trap 'errorMsg' EXIT

../plot.py -sv plot1.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres"
../plot.py -sv plot2.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres" "pc=bjacobi"
../plot.py -sv plot3.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres" -pc "geneo*ASM" -fe "overlap=1" "offloadE"
../plot.py -sv plot4.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres" -pc "geneo*SORAS" -fe "overlap=1" "offloadE" "tau=0.20" -t "1.e-04" --plot2D
../plot.py -sv plot5.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres" -pc "geneo*SORAS" -fe "overlap=1" "pc=geneo0SORAS" "tau=0.10" -a1 readInp -a2 setUpSolve -a3 itSolve
