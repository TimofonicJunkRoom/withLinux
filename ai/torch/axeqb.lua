require 'torch'
require 'nn'
local log = require 'lumin_log.lua'

--@reference http://torch.ch/docs/five-simple-examples.html

log.warn('Solve Ax=b matrix equation')
local lr = 0.1 --> learning rate

log.info('data')

local A = torch.Tensor():eye(5,5)
local b = torch.Tensor():ones(5,1) * 3

print(A)
print(b)

log.info('net')
--local x = nn.Sequential():add(nn.Linear(5, 1, false))
local x = nn.Linear(5, 1, false)

log.info('loss:MSE')
local crit = nn.MSECriterion()
--local crit = nn.AbsCriterion() -- NOTE: not easy to converge

log.info('solve')
local loss_initial = 0
local losses = {}
local iters = {}
for i = 1, math.huge do
   log.info('iter '..tostring(i))

   --forward
   local netout = x:forward(A)
   -- print(netout)
   local loss = crit:forward(netout, b)
   print('  loss = ' .. tostring(loss))
   table.insert(losses, loss)
   table.insert(iters, iter)
   if loss_initial == 0 then loss_initial = loss end

   --backward
   x:zeroGradParameters()
   local critback = crit:backward(netout, b)
   x:backward(A, critback)

   --update
   x:updateParameters(lr)

   --detection
   --if loss < 1e-7 then break end
   if loss < 1e-9 then break end
   if loss > 20 * loss_initial then
      log.error('Loss explosion, initial='
        ..tostring(loss_initial)..' Current='..tostring(loss))
      os.exit(1)
   end
end

log.info('converged, dump solution')
--print(x:getParameters())
assert(x.bias == nil)
print(x.weight)

log.info('draw picture')
losses = torch.Tensor(losses)
require 'gnuplot'
--gnuplot.figure(1)
--gnuplot.title('CG loss minimisation over time')
--gnuplot.plot(losses)

gnuplot.pdffigure('junk.pdf')
gnuplot.plot({'MSE',  losses, '-'})
gnuplot.xlabel('Iteration')
gnuplot.ylabel('Loss')
gnuplot.plotflush()
