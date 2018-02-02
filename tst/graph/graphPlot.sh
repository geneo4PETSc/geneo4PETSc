#!/bin/bash -eu

function errorMsg {
  if [[ "$?" -ne "0" ]]; then echo "ERROR: check *.log"; fi
}
trap 'errorMsg' EXIT

../plot.py -sv plot1.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres"
../plot.py -sv plot2.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres" "pc=mg"
../plot.py -sv plot3.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres" -pc "geneo*ASM" -fe "overlap=0"
../plot.py -sv plot4.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres" -pc "geneo*SORAS" -fe "overlap=0" "optim=0.02" -t "1.e-04" --plot2D
../plot.py -sv plot5.png -sg size=10 -wk size=5 -fi "metis=dual" "ksp=gmres" -pc "geneo*SORAS" -fe "overlap=0" "pc=geneo2HSORAS" "pc=geneo2ESORAS" "optim=0.00" -a1 estimDimE -a2 realDimE -a3 nbIt
