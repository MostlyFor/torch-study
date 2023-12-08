import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F


np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)



I, H, O = 1, 10 ,1
W1, b1 = Variable(np.random.randn(I, H)), Variable(np.zeros(10))
W2, b2 = Variable(np.random.randn(H, O)), Variable(np.zeros(1))

def predict(x):
    y = F.linear_simple(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = F.linear_simple(y, W2, b2)
    return y


lr = 1 # lr
iters = 100000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()
    
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    
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