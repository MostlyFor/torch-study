import numpy as np
from dezero.core import Function
from dezero.core import as_variable
from dezero import utils

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


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1-y**2)
        return gx
    
def tanh(x):
    return Tanh()(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x : np.array):
        self.x_shape = x.shape # 원래 형상 x_shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy : np.array): # gy : Variable
        return reshape(gy, self.x_shape) # gradient는 원래 형상이 됨
    

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis = self.axis, keepdims = self.keepdims) # y.shape 유지
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims) # numpy 관련 이슈 
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x):
    return Sum()(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable()
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
    

def sum_to(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gw = matmul(x.T, gy)
        return gx, gw
    
def matmul(x,W):
    return MatMul()(x, W)

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb
    
def linear(x, W, b=None):
    return Linear()(x, W, b)

class MeanSquaredError(Function):
    def forward(self, x0, x1): # 예측값, 정답값
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. /len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(y_pred, y):
    return MeanSquaredError()(y_pred, y)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx
    
def exp(x):
    return Exp()(x)

def linear_simple(x, W , b= None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t+b
    t.data = None # t 데이터는 더이상 필요가 없음 + 역전파 시에도
    return y

class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)