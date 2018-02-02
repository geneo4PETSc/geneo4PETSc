#!/bin/bash -eu

echo ""
echo "C++ check."
echo ""
cppcheck -v --std=c++11 --enable=all --suppress=cstyleCast --error-exitcode=1 --includes-file=/usr/local/include src/*.cpp

echo ""
echo "Bash script check."
echo ""
shellcheck prq.sh tst/*/*.sh

echo ""
echo "Python script check."
echo ""
pylint --variable-rgx="[a-z]*([A-Z][a-z]*)*" --function-rgx="[a-z]*([A-Z][a-z]*)*" --argument-rgx="[a-z]*([A-Z][a-z]*)*" --class-rgx="[a-z]*([A-Z][a-z]*)*" --class-attribute-rgx="[a-z]*([A-Z][a-z]*)*" --method-rgx="[a-z]*([A-Z][a-z]*)*" --attr-rgx="[a-z]*([A-Z][a-z]*)*" --max-locals=20 --max-line-length=140 tst/*.py | awk 'BEGIN{rate = 0;} {print $0; if ($2 == "code" && $5 == "rated") {split($7, tokens, "/"); rate=tokens[1];}} END{print "rate = " rate; if (strtonum(rate) < 9.75) {print "KO - rate regression"; exit(1);};}'
