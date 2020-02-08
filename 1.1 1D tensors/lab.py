import torch
import numpy
import pandas
import matplotlib.pyplot as plotter

#Params: [{vector, color, name}]
def plotVec(vectors):
  ax = plotter.axes()

  for vec in vectors:
    ax.arrow(0, 0, *vec["vector"], head_width = 0.05, color = vec["color"], head_length = 0.1)
    plotter.text(*(vec["vector"] + 0.01), vec["name"])
  
  plotter.ylim(-2, 2)
  plotter.xlim(-2, 2)
  plotter.savefig('plot.png')


int_tensor = torch.tensor([0, 1])
print('dtype: ', int_tensor.dtype)
print('type: ', int_tensor.type())

float_tensor = torch.tensor([0.0, 1.0, 2.0, 3.6])
print('dtype: ' , float_tensor.dtype)
print('type: ', float_tensor.type())

n_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
print('type: ', n_float_tensor.type())

old_int_tensor = int_tensor
n_float_tensor = old_int_tensor.type(torch.FloatTensor)
print('type: ', n_float_tensor.type())

float_tensor.view(4, 1)