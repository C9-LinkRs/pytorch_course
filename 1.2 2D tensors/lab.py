import torch
import numpy

tensor_c = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
tensor_c[1:3, 1] = 0
print('changed tensor: ', tensor_c)
tensor_c[1:3, 2] = 0
print('changed tensor:  ', tensor_c)

print('--------------------------------')

x = torch.tensor([[1, 0], [0, 1]])
y = torch.tensor([[2, 1], [1, 2]])
print('sum: ', x+y)
print('sub: ', x-y)
print('hadamard mult: ', x*y)
print('scalar mult 2*x', 2*x)

print('--------------------------------')
a = torch.tensor([[0, 1, 1], [1, 0, 1]])
b = torch.tensor([[1, 1], [1, 1], [-1, 1]])
print('matrix mult: ', torch.mm(a, b))