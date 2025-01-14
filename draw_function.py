# 绘制 DynamicSAM 的 loss 和 rho 对应关系的分段函数图像
import matplotlib.pyplot as plt

x1 = [0, 0.01]
y1 = [0.3, 0.3]
x2 = [0.01, 0.1]
y2 = [0.2, 0.2]
x3 = [0.1, 0.5]
y3 = [0.1, 0.1]
x4 = [0.5, 1.0]
y4 = [0.05, 0.05]
x5 = [1.0, 1.5]
y5 = [0.0, 0.0]
    
plt.plot(x1, y1, label='rho=0.3, if 0.0 <= loss <= 0.01', color='b')
plt.plot(x2, y2, label='rho=0.2, if loss <= 0.1', color='r')
plt.plot(x3, y3, label='rho=0.1, if loss <= 0.5', color='g')
plt.plot(x4, y4, label='rho=0.05, if loss <= 1.0', color='y')
plt.plot(x5, y5, label='rho=0.0, if loss > 1.0', color='c')

plt.plot(0.01, 0.2, color='r', marker='o', markerfacecolor='none', markersize=3)
plt.plot(0.1, 0.1, color='g', marker='o', markerfacecolor='none', markersize=3)
plt.plot(0.5, 0.05, color='y', marker='o', markerfacecolor='none', markersize=3)
plt.plot(1.0, 0.0, color='c', marker='o', markerfacecolor='none', markersize=3)

plt.xlabel('Loss')
plt.ylabel('rho')
plt.title('Loss to rho')
plt.legend()
plt.grid(True)

plt.savefig('loss_to_rho.png')
