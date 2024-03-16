#                        start   end   stepsize   Algorithm   days   processes    selection stragegy
python -u src/test.py      10     20      1        ShortestQueue      70        4             slowest        2>&1 | tee out.txt
