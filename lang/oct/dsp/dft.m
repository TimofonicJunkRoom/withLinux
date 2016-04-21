function y = dft (x)

% argument
len = length(x);

ret = zeros(1, len);
for k = 1:len
    xk = 0;
    for n = 1:len
        xk = xk + x(n) * exp(j * (k-1) * (n-1) * (2 * pi)/ len);
    end
    ret(k) = xk;
end
y = ret;

end