import numpy as np
from dezero.core import Function

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs
        dydx = cos(x)
        gx = dydx * gy
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x = self.inputs
        dydx = np.sin(x)
        gx = dydx * gy
        return gx
    
def cos(x):
    return Cos()(x)
    