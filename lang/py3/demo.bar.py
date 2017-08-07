def barX(colorcode):
    return lambda x,xmax,width: print('{:>4.0%}'.format(x/xmax)+\
        '|'+'\x1b[{};1m'.format(colorcode)+'*'*round(width*x/xmax)+\
        ' '*round(width-width*x/xmax)+'\x1b[;m'+'|')
barG = barX('32')

import random
import time
import os

for i in range(1000):
    barG(random.random(), 1., os.get_terminal_size().columns-6)
    time.sleep(0.05)
