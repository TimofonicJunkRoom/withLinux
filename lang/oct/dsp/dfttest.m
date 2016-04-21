a = [ 1 1 1 1 ]

dfta_ = fft(a)
dfta = dft(a)
err = sum(abs(dfta_ - dfta))

y = dft(a)
yy = idft(y)
err2 = sum(abs(a - yy))