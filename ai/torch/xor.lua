-- DLBook, Bengio, chapter 6
require 'torch'
require 'nn'
local log = require 'lumin_log'

log.debug('init data')
local x = torch.Tensor():ones(4,2)
x[1][1] = 0
x[1][2] = 0
x[2][1] = 0
x[3][2] = 0
print(x)
local t = torch.Tensor():ones(4,1)
t[1][1] = 0
t[4][1] = 0
print(t)

log.debug('init model')
local mlp = nn.Sequential()
mlp:add(nn.Linear(2, 2, true))
mlp:add(nn.ReLU())
mlp:add(nn.Linear(2, 1, false))

log.debug('init crit')
local crit = nn.MSECriterion()

log.debug('training')
for i = 1, 1000 do
   local y = mlp:forward(x)
   --print(y)
   local loss = crit:forward(y, t)
   print(loss)
   local dloss = crit:backward(y, t)
   mlp:zeroGradParameters()
   mlp:backward(x, dloss)
   mlp:updateParameters(0.05)
end
print(mlp:forward(x))

log.debug('use the anwser')
print(mlp:parameters())
mlp:parameters()[1]:ones(2,2)
mlp:parameters()[2]:zeros(2)
mlp:parameters()[2][2] = -1
mlp:parameters()[3]:ones(1, 2)
mlp:parameters()[3][1][2] = -2
print(mlp:parameters())
print(mlp:forward(x))
