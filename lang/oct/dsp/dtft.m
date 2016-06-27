%!/usr/bin/matlab
% Zhou Mo 2016
clear
%% part 1, 2.28
x = [ -.5 0 .5 1 ]; % -1:2
h = [ 1 1 1 ]; % -2:0
y = conv(x, h)
figure;
stem([-3:2],y);
pause;
clear;
%% part 2, 2.25
x = [ 1 2 3 4 2 1];
a = [ 1 .5 ];
b = [ 1 0 2 ];
y = filter(b, a, x)
stem(y)
clear
%% part 3
n = 0:100;
w = -pi:pi/720:pi;
x = cos(pi*n/2);
X = x * exp(-j * n' * w)
y = x .* exp(j * (pi/4) * n .* x);
w0 = pi/4;
Y1 = y * exp(-j * n' * w);
Y2 = x * exp(-j * n' * (w-w0));
figure
stem(Y1)
stem(Y2)
