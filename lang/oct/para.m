% simple parallel computing in matlab
% octave parfor is just a stub
% use matlab if really want parallelization

sizemax=300;

disp('serial');
tic

c1 = 1;
for i = 1:sizemax
  c1 = c1+max(eig(rand(i,i)));
end

toc

% eval this line if using matlab
% matlabpool open; % deprecated in matlab R2015a
% parpool; % new pool creator

disp('parallel');
tic

c2 = 1;
parfor i = 1:sizemax
  c2 = c2+max(eig(rand(i,i)));
%endparfor
% use end if matlab
end

toc
