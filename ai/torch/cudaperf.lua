#!/usr/bin/th
-- cuda performance tester
-- https://github.com/torch/cutorch/issues/497

require 'torch'

local huge = 2500

print 'CPU mode reference'
local weight = torch.randn(256, 4096)
local timer = torch.Timer()
for iter = 1, huge do
   local data = torch.rand(4096, 8) -- batchsize 8
   local res = weight * data
end
print(timer:stop():time())

require 'cutorch'
print 'CUDA mode 1 -- cudamemcpy every iteration by mistake'

weight = weight:float() -- deliberate, emulating negligence
timer = timer:new()
for iter = 1, huge do
   local data = torch.rand(4096, 8) -- batchsize 8
   local res = torch.mm(weight:cuda(), data:cuda())
end
print(timer:stop():time())

print 'CUDA mode 2 -- normal'
weight = weight:cuda()
timer = timer:new()
for iter = 1, huge do
   local data = torch.rand(4096, 8):cuda() -- batch 8
   local res = torch.mm(weight, data)
end
print(timer:stop():time())

print 'CUDA mode 3 -- pre-create cudatensor, reducing sys time'
timer = timer:new()
weight = weight:cuda()
local mydata = torch.CudaTensor()
local myweight = torch.CudaTensor()
local myres = torch.CudaTensor()
for iter = 1, huge do
   local data = torch.rand(4096, 8)
   myweight:resize(weight:size()):copy(weight)
   mydata:resize(data:size()):copy(data)
   myres = torch.mm(myweight, mydata)
end
print(timer:stop():time())
