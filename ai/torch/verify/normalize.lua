require 'torch'
require 'nn'

a = torch.rand(10) --torch.Tensor({1,2,3,4,5})
delta = torch.ones(a:size())

-- control group
l = nn.Normalize(2)
b = l:forward(a)
ga = l:backward(a, delta)
print(a)
print(b)
print(ga)

-- experimental group
N = a:norm()
b_ = a / N
ga_ = delta / N - ( a * torch.dot(a, delta) )/(N*N*N)
print(a)
print(b_)
print(ga_)

-- compare
print((ga_ - ga):norm())
