import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
import dezero.layers as L



np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)



l1 = L.Linear(10)
l2 = L.Linear(1)


def predict(x):
    y = l1(x)
    y = F.sigmoid_simple(y)
    y = l2(y)
    return y


lr = 0.2 # lr
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()
    
    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    
    if i % 1000 == 0:
        print(loss)


y_pred = predict(x)


plt.scatter(x, y, color = 'blue')  # 산점도
print(y.shape)
print(y_pred.shape)
plt.scatter(x, y_pred.data, color = 'red') # 예측값

# 그래프에 레이블 추가
plt.xlabel('X')
plt.ylabel('Y = sinX')
plt.legend()  # 범례 표시

plt.show()