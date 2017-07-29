#!/usr/bin/th
-- cuda performance tester
-- https://github.com/torch/cutorch/issues/497

require 'torch'
require 'cutorch'

local huge = 2500
local weight = torch.randn(256, 4096):cuda()

print 'CUDA mode 2 -- normal'

cutorch.synchronize();
local timer = torch.Timer()

for iter = 1, huge do
   local data = torch.rand(4096, 8):cuda() -- batch 8
   local res = torch.mm(weight, data)
end

cutorch.synchronize();
print(timer:stop():time())

print 'CUDA mode 3 -- pre-create cudatensor, reducing sys time'

local mydata = torch.CudaTensor():resize(4096, 8)
local myweight = torch.CudaTensor():resize(weight:size()):copy(weight)
local myres = torch.CudaTensor():resize(256, 8)

cutorch.synchronize();
timer = timer:new()

for iter = 1, huge do
   local data = torch.rand(4096, 8) -- batch 8
   mydata:copy(data)
   torch.mm( myres, myweight, mydata ) -- C <- A * B
end

cutorch.synchronize();
print(timer:stop():time())
