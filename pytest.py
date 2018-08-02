import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import  numpy as np
print(np.ones([5,5])/np.pi)
m = nn.Softmax(dim=1)
input = autograd.Variable(torch.randn(2, 3))
output = m(input)
a,b = output.max(1)
print(input)
print(output)
print(a,b)