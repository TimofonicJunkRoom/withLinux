%!/usr/bin/octave
% FFT implementation
% 2016 Apr 19, Zhou Mo

function X = myfft(x)

%% stage 0, initialize

   [I, J] = size(x);
   X = [];

%% stage 1, argument check

   % I should be 1
   if I ~= 1
      disp('invalid input');
      return;
   end

   % J should be 2^m
   j = J;
   while j > 1
      j = j / 2;
   end
   if j ~= 1
      disp('invalid input');
      return;
   end

%% stage 2, map sequence

   y = fftmap(x);
   y = rot90(eye(length(y))) * y';
   y = y'; % e.g. 04261537

%% stage 3, reduce results

   X = fftreduce(y);
   
   return;
end
