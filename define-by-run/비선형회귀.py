import numpy as np
from dezero import Variable, Model, MLP
import dezero.layers as L
import dezero.functions as F
import matplotlib.pyplot as plt
from dezero import optimizers

# 학습 데이터 생성
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

# 하이퍼 파라미터 설정
lr = 0.2
iters = 10000


model = MLP([10, 1])
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)
        

for i in range(iters):
    
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    
    model.cleargrads()
    loss.backward()
    
    optimizer.update()
    
    if i % 1000 == 0:
        print(loss)


plt.scatter(x, y, color = 'blue')  # 산점도
plt.scatter(x, y_pred.data, color = 'red') # 예측값
plt.xlabel('X') # 그래프에 레이블 추가
plt.ylabel('Y = sinX')
model.plot(x)
plt.show()
