import q
import torch as th
from torch.autograd import Variable as V

a = V(th.rand(2,5), requires_grad=True)
print('a', a)
l = th.nn.Linear(5,3)
y = l.forward(a)
print('y', y)

y_ = th.mm(l.weight, a.t()).t() + l.bias.expand(2,3)
print('y_', y_)
print('||y_-y||_2', (y_-y).norm())

gy = th.rand(2,3)
l.zero_grad()
y.backward(gy)
ga = a.grad
print('ga', ga)
gb = l.bias.grad
print('gb', gb)
gw = l.weight.grad
print('gw', gw)

ga_ = th.mm(l.weight.data.t(), gy.t()).t()
print('ga_', ga_)
gb_ = gy.t().sum(1).t()
print('gb_', gb_)
gw_ = th.mm(gy.t(), a.data)
print('gw_', gw_)

p = lambda x,y: print((x-y).norm())
p(ga.data, ga_)
p(gw.data, gw_)
p(gb.data, gb_)

#q.d()
