require 'torch'
require 'nn'

-- prepare
local a = torch.rand(1,5)
local l = nn.SoftMax()

-- reference result
local y = l:forward(a)

-- softmax forward
local yhat = torch.exp(a):cdiv( torch.sum(torch.exp(a),2):expandAs(a) )

-- compare
print((y-yhat):norm())

local delta = torch.rand(1,5)
gy = l:backward(a, delta)
print(gy)
local jacob = torch.zeros(5,5)
for i = 1, 5 do
	for j = 1, 5 do
		if i == j then
			jacob[i][j] = y[1][i] - y[1][i]*y[1][i]
		else
			jacob[i][j] = - y[1][j]*y[1][i]
		end
	end
end
gyhat = delta * jacob:t()

print(gy, gyhat)
