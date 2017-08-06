# profiling your python program foobar.py

# 1. pip3 install gprof2dot

# 2. python3 -m cProfile -o xxx.pstats foobar.py

# 3. gprof2dot -f pstats xxx.pstats | dot -Tsvg foobar.prof.svg

# ... https://wiki.python.org/moin/PythonSpeed/PerformanceTips
