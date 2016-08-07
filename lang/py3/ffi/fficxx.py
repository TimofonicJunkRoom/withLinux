print("\x1b[31;1mPython3 Starts to interpret from here\x1b[m")
# https://docs.python.org/3/library/ctypes.html
# http://starship.python.net/crew/theller/ctypes/tutorial.html
import ctypes

kernel = ctypes.CDLL("./cxxkernel.so")

#print(kernel._Z6kernelPc)
#print(kernel._Z6kernelPc("hello"))

print(kernel._Z6kernelNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE("hello"))
