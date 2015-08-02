%!/usr/bin/octave
function output = conv (matrix)

imgsize = size(matrix);
conv_mask = [ 1 0 1; 0 1 0; 1 0 1 ];

m = [];
for i = 1:imgsize(1)-2
	for j = 1:imgsize(2)-2
		m(i,j) = sum(sum( conv_mask .* matrix(i:i+2,j:j+2) ));
	end
end

output = m;
