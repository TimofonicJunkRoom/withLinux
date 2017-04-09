require 'nn'
require 'mydropout_lua'

-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)
local percentage = 0.5
local input = torch.Tensor(ini,inj,ink):zero()
local module = nn.MyDropout(percentage)

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end
