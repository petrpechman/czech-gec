input_path=$1
# cat ../../data/akces-gec/dev/dev.all.m2  | grep "^..*$" | sed 's|^S|\nS|' | sed 1d > input.m2
cat ${input_path}  | grep "^..*$" | sed 's|^S|\nS|' | sed 1d > input.m2
errant_m2 -auto input.m2 -out mid.m2
retag -i mid.m2 -o out.m2 -c