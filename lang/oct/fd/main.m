% finite difference equations
% Copyright (C) 2016 Zhou Mo

% N should be > 2
N = 50;

% boundary conditions

% condition 1
V = zeros(N,N);
V(1,:)=100;

% condition2
%omega = (2*pi) / N;
%omega = omega * [ 1: N ];
%omega = sin(omega)
%size(omega)
%size(V(1,:))
%V(1,:) = omega;
%V(:,1) = omega;
%V(N,:) = -omega;
%V(:,N) = -omega;

% condition 3
a = ones(2,N/2);
a(1,:) = 0;
a = reshape(a, 1, N);
V(1,:) = a;

% condition4
V(1,:) = log(1:N);

% condition5
V(1,:) = [1:N] / N;

% construct A and b in 'Ax = b'
A = zeros( (N-2)^2 , (N-2)^2 );
b = zeros(       1 , (N-2)^2 );
for i = 2:N-1
   for j = 2:N-1
      n = (i-2)*(N-2)+(j-1);
      % up
      if (i-1) == 1
          b(1, n) = b(1, n) - V(i-1, j);
      else
          A(n, n-(N-2)) = 1;
      end
      % down
      if (i+1) == N
          b(1, n) = b(1, n) - V(i+1, j);
      else
          A(n, n+(N-2)) = 1;
      end
      % left
      if (j-1) == 1
          b(1, n) = b(1, n) - V(i, j-1);
      else
          A(n, n-1) = 1;
      end
      % right
      if (j+1) == N
          b(1, n) = b(1, n) - V(i, j+1);
      else
          A(n, n+1) = 1;
      end
      % center
      A(n, n) = -4;
   end
end

prettyA = -A;
prettyb = -b.';

solution = prettyA\prettyb;

s = reshape(solution, N-2, N-2).';
V(2:N-1,2:N-1) = s;

[X,Y] = meshgrid(1:N,1:N);
surf(X,Y,V);
