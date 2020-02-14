import torch
import matplotlib.pylab as plt
import torch.nn.functional as F

x = torch.tensor(2.0, requires_grad=True)
print('tensor: ', x)

y = x ** 2
print('function evaluated on tensor x: ', y)

y.backward()
print('tensor derivate of y: ', x.grad)

print('y function backward graph operator:', y.grad_fn)

############################################################################

u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(3.6, requires_grad=True)
f = u ** 3 + v ** u + 1

print('function evaluated on tensor u and v: ', f)

f.backward()

print('tensor partial derivate of f respect to u:', u.grad)
print('tensor partial derivate of f respect to v: ', v.grad)

print('f function backward graph operator: ', f.grad_fn)

############################################################################

z = torch.linspace(-10, 10, 10, requires_grad=True)
k = z ** 2

print('function evaluated on tensor z: ', k)

k.backward(torch.ones(10)) # or k.backward(torch.ones_like(z))

print('tensor derivate of k with tensor shape: ', z.grad)

z.grad.zero_() # set grad to zero
h = torch.sum(z ** 2)
h.backward()

print('tensor derivate of k with sum trick: ', z.grad)

plt.plot(z.detach().numpy(), k.detach().numpy(), label='function')
plt.plot(z.detach().numpy(), z.grad.numpy(), label='derivate')
plt.xlabel('x')
plt.legend()
plt.savefig('plot x^2.png')

############################################################################

x = torch.linspace(-3, 3, 100, requires_grad=True)
y = F.relu(x)
y.backward(torch.ones_like(x))

plt.clf() # clear figure
plt.plot(x.detach().numpy(), y.detach().numpy(), label='function')
plt.plot(x.detach().numpy(), x.grad.numpy(), label='derivate')
plt.xlabel('x')
plt.legend()
plt.savefig('plot relu.png')