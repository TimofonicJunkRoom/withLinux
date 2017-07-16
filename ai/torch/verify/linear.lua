require 'torch'
require 'nn'

x = torch.rand(1, 10)
l = nn.Linear(10, 3) -- insize, outsize
w = l.weight
b = l.bias
gy = torch.ones(1, 3)

--print(w)
print(w:size())
--print(b)
print(b:size())

--- control group
y = l:forward(x)
l:zeroGradParameters()
gx = l:backward(x, gy)
l:accGradParameters(x, gy)
gb = l.gradBias
gw = l.gradWeight

--- experiment group
yy = x * w:t() + b
ggx = gy * w
ggb = gy:t() * 2.
ggw = (x:t() * gy):t() * 2.

print(yy)
print(y)

print(gx)
print(ggx)

print(gb)
print(ggb)

print(gw)
print(ggw)

-- FIXME: the results differ?
