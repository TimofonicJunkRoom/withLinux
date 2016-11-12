#!/usr/bin/julia

P = []

for i = 1:1000
   if isprime(i)
      P = [ P; i ]
   end
end

for i = 1:length(P)
   print(P[i], ' ')
   if i != 1 && i % 8 == 0
      print('\n')
   end
end
