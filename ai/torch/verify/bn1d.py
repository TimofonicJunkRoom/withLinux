import torch as th
from torch.autograd import Variable

N,D = 3,5

x = Variable(th.rand(3,5), requires_grad=True)
#x = Variable(th.ones(N,D), requires_grad=True)
l = th.nn.BatchNorm1d(D, momentum=.0)
l.train()
y = l(x)
gamma = l.weight
beta = l.bias
gy = Variable(th.rand(3,5), requires_grad=False)
#gy = Variable(th.ones(3,5), requires_grad=False)
print('gy', gy)

print('x', x)
print('y', y)

mu = x.sum(0)/N # size 5
#print(mu)
var = ((x - mu.expand(N,D))**2).sum(0)/N # size 5
xhat = (x - mu.expand(N,D))/th.sqrt(var.expand(N,D) + 1e-5) # size 3x5
vy = xhat.mul(gamma.expand(N,D)) + beta.expand(N,D)
print('vy', vy)
print('vy-y', (vy-y).norm())

l.zero_grad()
y.backward(gy)

vgbeta = gy.sum(0)
print('gbeta', beta.grad)
print('vgbeta', vgbeta)

vggamma = xhat.mul(gy).sum(0)
print('ggamma', gamma.grad)
print('vggamma', vggamma)
print('norm', (gamma.grad - vggamma).norm())

print('gx', x.grad)
# https://arxiv.org/pdf/1502.03167.pdf
# The backward pass for gradInput is complicated.
vgxhat = gy.mul(gamma.expand(3,5))
vgvar = (vgxhat.mul(x-mu).mul(-1/2).mul(th.pow(var+1e-5, -3/2))).sum(0)
vgmu = (vgxhat.mul(-1).mul(th.pow(var+1e-5, -1/2))).sum(0) + \
        vgvar.mul(1/3).mul( (2*mu - 2*x).sum(0) )
vgx = vgxhat.mul(th.pow(var+1e-5, -1/2)) + \
        vgvar.mul(2/3).mul(x-mu) + \
        vgmu/3
print('vgx', vgx)
