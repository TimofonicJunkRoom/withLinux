print("\x1b[31;1mPython3 Starts to interpret from here\x1b[m")
# https://docs.python.org/3/library/ctypes.html
# http://starship.python.net/crew/theller/ctypes/tutorial.html
import ctypes

kernel = ctypes.CDLL("./libkernel.so")

''' without this line this program will end up with "Illegal Instruction" '''
#print(kernel._ZN6kernel6kernel17h6daf402dcf741961E)

print(kernel._ZN6kernel6kernel17h6daf402dcf741961E("hello"))

