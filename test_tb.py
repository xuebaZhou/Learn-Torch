from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
# image_path = "data/hymenoptera_data/train/ants_image/0013035.jpg"
image_path = "data/hymenoptera_data/train/bees_image/98391118_bdb1e80cce.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

# writer.add_image("test", img_array, 2, dataformats='HWC')  # 这里tag没有变，因此就显示在一个地方，通过滑块进行滑动
# 从PIL到numpy，需要在add_image()中指定shape中每一个数字/维表示的含义,这里要是维度是(高，宽，通道)，需要加上 dataformats='HWC'
writer.add_image("train", img_array, 1, dataformats='HWC')
# y = x
for i in range(100):
    writer.add_scalar("y = x^2", i * i, i)
    # 要是存在之前的函数，这里就会进行一个拟合，要是不想这样，将生成的文件删除，重新运行即可

writer.close()
# 终端运行 tensorboard --logdir=D:\计算机汇总\Pytorch\learn-torch\pythonProject1\logs  --port=6007 进行打开
