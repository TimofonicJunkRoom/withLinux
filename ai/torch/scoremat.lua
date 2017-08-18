require 'torch'

local testtimes = 100

local x = torch.randn(251, 4096)
local v = torch.randn(251, 4096)

print('one-pass calculation')
t = torch.Timer()
for i = 1, testtimes do
   local s = torch.mm(x, v:t()) -- 251x251
end
print(t:stop():time())

print('one-by-one calculation')
t = torch.Timer()
for i = 1, testtimes do
   local nsamples = x:size()[1]
   local s = torch.zeros(nsamples, nsamples)
   for j = 1, nsamples do
	  for k = 1, nsamples do
		 s[j][k] = torch.dot(x[j], v[k])
	  end
   end
end
print(t:stop():time())

--[[

I5-2520M, OMP_NUM_THREADS=2

one-pass calculation	
{
  real : 1.622526884079
  sys : 0.291489
  user : 6.073291
}
one-by-one calculation	
{
  real : 41.113063812256
  sys : 0.240375
  user : 41.170221
}

I7-6900K, OMP_NUM_THREADS=4

one-pass calculation
{
  real : 0.66684794425964
  sys : 0.152
  user : 2.516
}
one-by-one calculation
{
  real : 21.899141073227
  sys : 0.228
  user : 21.924
}

--]]
