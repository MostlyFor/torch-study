import numpy as np

class Optimizer:
    def __init__(self):
        self.target = None # 최적화할 모델
        self.hooks = []
    
    def setup(self, target):
        self.target = target
        return self
        
    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        
        # 전처리
        for f in self.hooks:
            f(params)
        
        for param in params:
            self.update_one(param)
        
    def update_one(self, param):
        raise NotImplemented
    
    def add_hook(self, f): # 가중치 감소 및 기울기 클리핑 같은 기법 가능
        self.hooks.append(f)
        

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
        
    def update_one(self, param):
        param.data -= self.lr * param.grad.data
        
class MomentumSGD(Optimizer):
    def __init__(self, lr = 0.01, momentum = 0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {} # 파라미터들의 속도 저장
    
    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum # \alpha * v
        v -= self.lr * param.grad.data
        param.data += v
        
        
        