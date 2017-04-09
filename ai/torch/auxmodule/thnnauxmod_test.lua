
require 'torch'
require 'nn'

--require 'libthnnauxmod'
--require 'mythreshold'
--
require 'thnnauxmod'

print(thnnauxmod)
print(thnnauxmod.MyThreshold)

local layer = thnnauxmod.MyThreshold()
print(layer)

local a = torch.rand(10,10)-0.5
print(a)

local output = layer:forward(a)
print(output)

local gradOutput = torch.rand(10,10)-0.5
print(gradOutput)

local gradInput = layer:backward(output, gradOutput)
print(gradInput)
