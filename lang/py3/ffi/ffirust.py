print("\x1b[31;1mPython3 Starts to interpret from here\x1b[m")
# https://docs.python.org/3/library/ctypes.html
# http://starship.python.net/crew/theller/ctypes/tutorial.html
import ctypes

kernel = ctypes.CDLL("./libkernel.so")

try:
  ''' without this line this program will end up with "Illegal Instruction" '''
  print(kernel._ZN6kernel6kernel17h6daf402dcf741961E) # s: &'static str
  msg = "Hello rust library".encode("ascii")
  print(kernel._ZN6kernel6kernel17h6daf402dcf741961E(msg)) # this matches rust function definition but doesn't work
  print(kernel._ZN6kernel6kernel17h6daf402dcf741961E(ctypes.c_char_p(u"hello".encode()), ctypes.c_size_t(5)))
  print(kernel._ZN6kernel6kernel17h6daf402dcf741961E(ctypes.c_char_p(u"hello".encode("UTF-8")), 5))
  print(kernel._ZN6kernel6kernel17h6daf402dcf741961E(u"hello".encode("UTF-8"), 5))
  print(kernel._ZN6kernel6kernel17h6daf402dcf741961E(bytes(u"hello".encode("UTF-8")), 5))
except:
  print("_ZN6kernel6kernel17h6daf402dcf741961E not working")

try:
  print(kernel._ZN6kernel6kernel17h081d6f856694018aE) # s: &String
  print(kernel._ZN6kernel6kernel17h081d6f856694018aE(ctypes.c_char_p(u"hello".encode())))
  print(kernel._ZN6kernel6kernel17h081d6f856694018aE(u"hello".encode()))
except:
  print("_ZN6kernel6kernel17h081d6f856694018aE not working")

try:
  print(kernel._ZN6kernel6kernel17h48d014a057321023E) # s: &str
  print(kernel._ZN6kernel6kernel17h48d014a057321023E(ctypes.c_char_p(u"hello".encode())))
  print(kernel._ZN6kernel6kernel17h48d014a057321023E(ctypes.c_char_p(u"hello")))
except:
  print("_ZN6kernel6kernel17h48d014a057321023E not working")
