import pytools.prefork as pf

print pf.call_capture_stdout(["nvcc", "--version"])
pf.enable_prefork()
from time import sleep
print "NOW"
sleep(17)

print pf.call_capture_stdout(["nvcc", "--version"])
