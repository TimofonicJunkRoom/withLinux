% circular convolution
function ret = circ (x, y, N)

c = conv(x, y)
len = length(c)


ret = zeros(1, N)
if len <= N
    ret(1:len) = c
    return
end
% len > N, reduce
for i = 1:len
    cur = mod(i, N)
    if cur == 0
        cur = N
    end
    ret(cur) = ret(cur) + c(i)
end

end
