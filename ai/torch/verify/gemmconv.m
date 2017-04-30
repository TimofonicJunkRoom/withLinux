
% regular 1d convolution
S = 0;
vsize = 10;
ksize = 3;
tsize = vsize + ksize - 1;

for i = 1:100
	% populate signal
	signal = rand(vsize, 1);
	signal_pad = postpad(signal, tsize);

	% populate kernel and trans mat
	k = rand(ksize, 1);
	krev = rot90(rot90(k));
	trans = zeros(tsize, tsize);
	for i = 1:tsize
		if i >= ksize
			trans(i,i-ksize+1:i) = krev(:);
		else
			trans(i,1:i) = krev(ksize-i+1:ksize);
		end
	end
	trans;

	% gemm
	res = trans * signal_pad;

	% reference
	ref = conv(signal, k);

	% comparison
	S = S + sum(abs(res - ref));
end
S
