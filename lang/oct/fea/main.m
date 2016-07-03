%% Finite Element Analysis example from coursebook
% Zhou Mo 2016

points = [
    0 2;
    1 2;
    2 2;
    0 1;
    1 1;
    2 1;
    0 0;
    1 0;
    2 0];

S = {};
S{1} = getexauv(points, 1, 4, 5);
S{2} = getexauv(points, 1, 2, 5);
S{3} = getexauv(points, 2, 3, 5);
S{4} = getexauv(points, 3, 5, 6);
S{5} = getexauv(points, 4, 5, 7);
S{6} = getexauv(points, 5, 7, 8);
S{7} = getexauv(points, 5, 8, 9);
S{8} = getexauv(points, 5, 6, 9);

SS = zeros(size(S{1}));
for i = 1 : 8
    SS = SS + S{i};
end
SS

%syms v5;
%V = [ 100, 100, 100, 0, v5, 0,  0, 0, 0 ].';
V = [ 100, 100, 100, 0, 25, 0,  0, 0, 0 ].';
SS*V
%SSV = [ SS V ];
%rref(SSV)
%V(5) = 25
%sum(SS*V)
%syms v1 v2 v3 v4 v5 v6 v7 v8 v9;
%V = [ v1 v2 v3 v4 v5 v6 v7 v8 v9 ].';
%Sol = solve(sum(SS*V), 'v1=100', 'v2=100', 'v3=100', 'v4=0', 'v6=0', 'v7=0', 'v8=0', 'v9=0')
