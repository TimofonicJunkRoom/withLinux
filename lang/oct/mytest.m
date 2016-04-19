a = 0:7;

target = fft(a);
my = myfft(a);

err = sum(abs(my - target))
