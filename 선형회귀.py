import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph

# 시드 설정
np.random.seed(0)

# 데이터 생성
x = Variable(np.random.rand(100, 1))
y = 5 + 2 * x + np.random.rand(100, 1)

W = Variable(np.zeros([1,1]))
b = Variable([1])

steps = 100
lr = 0.01

for _ in range(steps):
    W.cleargrad()
    b.cleargrad()
    pred = W * x + b
    error = F.mean_squared_error(pred,y)
    error.backward()
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    
print(W, b)
    
