% myfft internal module
% mapping util

% fftmap(a) = [ fftmap(a'(1:n/2 -1)), fftmap(a'(n/2:n)) ]

function y = fftmap(x)
   if length(x) == 1
      y = x;
   else % length(x) > 1
      % split
      x1 = [];
      x2 = [];
      for i = 1:length(x)
         if mod(i, 2) == 0
            x1 = [ x1, x(i) ];
         else
            x2 = [ x2, x(i) ];
         end
      end
      y = [ fftmap(x1), fftmap(x2) ];
   end
   return;
end
