require 'nn'

a = torch.rand(3,5)
l = nn.Sigmoid()

y = l:forward(a)
yhat = (1.+torch.exp(-a)):pow(-1)

print((y-yhat):norm())

delta = torch.ones(3,5)
gy = l:backward(y, delta)
gyhat = torch.cmul(y, 1.-y):cmul(delta)

print(gy)
print(gyhat)
