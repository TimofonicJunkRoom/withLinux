require 'nn'

a = torch.rand(3,5)
l = nn.Tanh()

y = l:forward(a)
yhat = (torch.exp(a) - torch.exp(-a)):cdiv(torch.exp(a) + torch.exp(-a))

print(y, yhat)
print((y-yhat):norm())

delta = torch.rand(3,5)
--delta = torch.ones(3,5)
gy = l:backward(a, delta)
gyhat = (1 -torch.pow(y, 2)):cmul(delta)

print(gy, gyhat)
