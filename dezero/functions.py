import numpy as np
from dezero.core import Function

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = cos(x) * gy
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = -sin(x) * gy
        return gx
    
def cos(x):
    return Cos()(x)


    