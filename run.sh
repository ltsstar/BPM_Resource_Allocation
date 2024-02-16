#                        start   end   stepsize   Algorithm   days   processes    selection stragegy
python -u src/test.py      1     1.1      0.1      MILP        1        1             fastest        2>&1 | tee out.txt
