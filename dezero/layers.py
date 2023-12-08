from dezero.core import Parameter
import numpy as np
import weakref
import dezero.functions as F

class Layer:
    def __init__(self):
        self._params = set()
        
    def __setattr__(self, name, value):
        if isinstance(value,Parameter): # Parameter -> 이름 추가
            self._params.add(name)
        super().__setattr__(name, value)
        
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplemented
    
    def params(self):
        for name in self._params:
            yield self.__dict__[name]
    
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
        


class Linear_old(Layer):
    def __init__(self, in_size, out_size, nobias = False, dtype = np.float32):
        super().__init__()
        
        I, O = in_size, out_size
        W_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1/I) # 가중치 초기화
        self.W = Parameter(W_data, name = 'W')
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(0, dtype = dtype), name = 'b')
            
    def forward(self, x):
        y = F.linear(x, self.W, self.b)
        return y


class Linear(Layer): # input_size 조절 x
    def __init__(self, out_size, nobias = False, dtype = np.float32, in_size = None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name = 'W')
        
        if self.in_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype = dtype), name = 'b')
        
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1/I)
        self.W.data = W_data
        
    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y
        