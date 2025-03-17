# import torch
# from torch.utils.data import Dataset
# from torch.utils.tensorboard import SummaryWriter
# from PIL import Image
#
# writer = SummaryWriter(log_dir='logs')
# for i in range(100):
#     writer.add_scalar("y=x", i, i)
#
# writer.close()

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 数据
semesters = ['大一上', '大一下', '大二上', '大二下']
scores = [96.85, 99.19, 97.72, 97.36]

# 创建一个新的图形
fig, ax = plt.subplots(figsize=(8, 6))

# 使用不同的颜色来区分气泡（这里假设所有点都使用相同的颜色）
color = 'blue'

# 绘制带有圆形标记的折线图
line, = ax.plot(semesters, scores, marker='o', color=color, linestyle='-', markersize=10)

# 设置y轴范围和刻度
ax.set_ylim(96, 100)
ax.set_yticks([96, 97, 98, 99, 100])  # 减少y轴的刻度数量

# 设置标题和标签
plt.title('四学期成绩变化图')
plt.xlabel('学期')
plt.ylabel('综合学分绩')

# 添加网格线
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)  # 更改网格线样式和透明度

# 设置x轴标签
ax.set_xticks(range(len(semesters)))
ax.set_xticklabels(semesters)

# 在图表右上角添加注释
ax.annotate('蓝色的线代表综合学分绩',
            xy=(semesters[-1], scores[-1]), xycoords='data',
            xytext=(+20, +50), textcoords='offset points', fontsize=12,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# 显示图表
plt.show()
