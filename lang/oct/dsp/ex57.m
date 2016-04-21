n = 0 : 7
N = length(n)

x1 = sin(pi * n / 2)

target = fft(x1)
my = myfft(x1)

err = sum(abs(target - my))