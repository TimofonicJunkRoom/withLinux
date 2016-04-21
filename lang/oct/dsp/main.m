% fft benchmark

x = rand(1, 1024);

disp ('dft ...');
tic;
X1 = dft(x);
t0 = toc

disp ('fft ...');
tic;
X2 = myfft(x);
t1 = toc

disp ('fft reference');
tic;
X3 = fft(x);
t2 = toc

disp ('');
disp(sprintf('myfft to dft acceleration ratio %f', t0/t1));
disp(sprintf('fft to dft acceleration ratio %f', t0/t2));
disp(sprintf('fft is %f times faster than myfft', t1/t2));
