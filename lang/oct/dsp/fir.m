% DSP homework, BPF FIR filter.
clear;
close all;
%% BPF FIR, window function design technique
% configure
% lower stopband 0.2 pi as 60db
% lower passband 0.3 pi rp 0.5db
% upper passband 0.6 pi rp 0.5db
% upper stopband 0.7 pi as 60db
% (a) window function selection
% (b) main design steps
% (c) verify tolerance
w_sl = 0.2 * pi;
w_su = 0.7 * pi;
w_pl = 0.3 * pi;
w_pu = 0.6 * pi;
alpha_s = 60; % db
r_p = 0.5; % db

%% design
% I II III IV all do, select I
% select blackman, stop band min decrease 74 db
delta_b = 0.1 * pi; % while delta_b of blackman window is 11pi/N rad
% 11 \pi / N <= 0.1 \pi --> N >= 110
N = 112;
Wn = [
    (w_sl + w_pl)/2,
    (w_pu + w_su)/2];
Wn = Wn ./ pi;
hd = fir1(N, Wn, blackman(N+1));
[h, w] = freqz(hd);
hdb = 20 * log10 ( abs(h) ./ abs(h(2)) );
plot(w/pi, hdb);

% pass band min decrease
mask_passband = (w > 0.3 * pi) .* (w < 0.6 * pi);
hdb_len = length(hdb);
alpha_stop = 20 * log10( max(hdb .* mask_passband)/hdb(ceil(hdb_len*0.3)) );
alpha_stop

% test filter
Fs = 100; % sampling freq
T = 1/Fs; % sample time
L = 100; % length
t = (0:L-1)*T; % time vector
signal1 = sin( 0.1 * pi * t);
signal2 = sin( 0.5 * pi * t);
signal3 = sin( 1 * pi * t);
y = signal1 + signal2 + signal3;
NFFT = 2^nextpow2(L);
Y = fft(y, NFFT)/L;
f = Fs/2 * linspace(0, 1, NFFT/2+1);
fw = f * 2 * pi;
plot(fw, 2*abs(Y(1:NFFT/2+1)));

figure;
plot(abs(fft(signal3)));
figure;
plot(abs(fft(conv(signal3, hd))));
%plot(linspace(-pi, pi, length(t)), abs(fft(signal1)));
