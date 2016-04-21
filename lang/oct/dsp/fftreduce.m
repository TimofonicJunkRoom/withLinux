% myfft internal helper
% fft reduction util

% fftreduce(x) = [ fftreduce(1st half), fftreduce(2nd half) ]

% x is reversed fftmap output
function y = fftreduce(x)
   if length(x) == 1
      y = x;
   else
      x1 = x(1:((length(x)/2))); % first half
      x2 = x(length(x)/2+1:length(x)); % seconf half
      x1 = fftreduce(x1);
      x2 = fftreduce(x2);

      p = 0:((length(x)/2)-1);
      W = exp(-j * (2*pi)/(length(x)));
      w = W .^ p;

      y = [ x1+x2.*w, x1-x2.*w ];
   end
   return;
end
