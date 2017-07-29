require 'torch'
require 'nn'

x = torch.rand(1, 10)
l = nn.Linear(10, 3) -- insize, outsize
w = l.weight
b = l.bias
gy = torch.rand(1, 3)

print(w:size())
print(b:size())

y = l:forward(x)
l:zeroGradParameters()
gx = l:backward(x, gy)
--l:accGradParameters(x, gy) -- backward contains accGradParameters, doubling this call will double the gradient
gb = l.gradBias
gw = l.gradWeight

yy = x * w:t() + b
ggx = gy * w
ggb = gy:t()
ggw = (x:t() * gy):t()

print('yy')
print(yy)
print(y)

print('gx')
print(gx)
print(ggx)

print('gb')
print(gb)
print(ggb)

print('gw')
print(gw)
print(ggw)

-- FIXME: the results differ?
