
require 'torch'

local max=600

print('begin double')
torch.setdefaulttensortype('torch.DoubleTensor')
local tm = torch.Timer()
for i = 1,max do
	local a = torch.rand(i, i)
	local b = torch.rand(i, i)
	local c = torch.mm(a, b)
end
tm:stop()
print('done', tm:time())

print('begin float')
torch.setdefaulttensortype('torch.FloatTensor')
local tm = torch.Timer()
for i = 1,max do
	local a = torch.rand(i, i)
	local b = torch.rand(i, i)
	local c = torch.mm(a, b)
end
tm:stop()
print('done', tm:time())

--[[ without setting OMP_NUM_THREADS / OMP_NUM_THREADS=4

begin double  2520M
  real : 6.1052939891815
begin float	
  real : 4.8923819065094

    OMP_NUM_THREADS=2 th cpubench.lua 

begin double	2520M
  real : 3.8089778423309
begin float	
  real : 2.7467200756073

begin double	6900K threads=4
real : 2.0488569736481
begin float	    6900K
real : 1.3556079864502

--]]
