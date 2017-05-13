% Game of Life

k = [ 1 1 1 ; 1 0 1; 1 1 1];
getScore = @(m)(conv2(m, k, 'same'));
iter = 0;

%m = [ 0 0 0; 1 1 1; 0 0 0 ]
%m = [ 0 1 0 0; 0 0 1 0; 1 1 1 0; 0 0 0 0]
m = rand(20,30)>0.5;
	printf('I: iteration %d\n\n', iter)
	for i = 1:size(m, 1)
		for j = 1:size(m, 2)
			if m(i,j) == 0
				printf('. ');
			else
				printf('o ');
			end
		end
		printf('\n');
	end

while true
	%%%  score = getScore(m);
	%%%  % non-empty cells
	%%%  survivor = (m.*score == 2) + (m.*score == 3);
	%%%  % empty cells
	%%%  newlife = (m==0).*(score == 3);
	%%%  % next epoch
	%%%  m = newlife + survivor;
	s = getScore(m);
	oldm = m;
	m = (m==0).*(s==3) + (m.*s == 2) + (m.*s == 3);
	% dump map
	iter = iter + 1;
	printf('I: iteration %d\n\n', iter)
	for i = 1:size(m, 1)
		for j = 1:size(m, 2)
			if m(i,j) == 0
				printf('. ');
			else
				printf('o ');
			end
		end
		printf('\n');
	end
	disp('')
	if norm(m - oldm) > 0
		pause
	else
		printf('Game over!');
	end
end
