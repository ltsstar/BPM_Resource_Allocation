#                    start   end   stepsize   Algorithm   days   processes
python -u src/test.py      1     1.1    0.1      MILP        365        1        2>&1 | tee out.txt
