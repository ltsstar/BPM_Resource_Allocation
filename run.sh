#                        start   end   stepsize   Algorithm   days   processes    selection stragegy problem
python -u src/test.py      1.0   1.1      0.01        Hungarian      365        4             slowest PO       2>&1 | tee out.txt
