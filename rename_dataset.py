import os

root_dir = "data/hymenoptera_data/train"
target_dir = "ants_image"
# 获取目标目录下的所有文件和子目录名称
img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split('_')[0]  # 按照_进行切割，得到的是ants
out_dir = "ants_label"
for i in img_path:
    file_name = i.split(".jpg")[0]  # 取出照片名
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)  # 写入的数据 这里就是写入 ants
