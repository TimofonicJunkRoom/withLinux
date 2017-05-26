require 'torch'
require 'nn'
local log = require 'lumin_log.lua'
local dataset = require 'dataset'

log.debug('set up network')
mlp = nn.Sequential()
inputs = 2
outputs = 1
hiddenunits = 25
mlp:add(nn.Linear(inputs, hiddenunits))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(hiddenunits, outputs))
print (mlp)

log.debug('populate data')
dataset.populate()

log.debug('check data availability')
print(dataset.size())
print(dataset[1][1])

log.debug('loss function')
crit = nn.MSECriterion()
print (crit)

log.debug('trainer')
trainer = nn.StochasticGradient(mlp, crit)
trainer.learningRate = 0.01
print (trainer)

log.debug('upgrade function')
function gradUpdate(mlp, x, y, crit, lr)
	local netout = mlp:forward(x)
	local err = crit:forward(netout, y)
	local gradcrit = crit:backward(netout, y)
	mlp:zeroGradParameters()
	mlp:backward(x, gradcrit)
	mlp:updateParameters(lr)
	return err
end

log.debug('train')

-- high level training
trainer:train(dataset)

-- low level training
for i = 1, dataset:size() do
	--log.info('iteration ' .. i)
	local d_err = gradUpdate(mlp, dataset[i][1], dataset[i][2], crit, 0.01)
	print(d_err)
end
