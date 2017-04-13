require 'torch'
require 'nn'

-- prepare
local a = torch.rand(3,5)
local l = nn.SoftMax()

-- reference result
local yhat = l:forward(a)

-- softmax forward
local y = torch.exp(a):cdiv( torch.sum(torch.exp(a),2):expandAs(a) )

-- compare
print((y-yhat):norm())
