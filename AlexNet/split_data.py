import os
import random
from shutil import copy,move

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)
        
file_path = 'data/ALL'
expression_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]

mkfile('data/train')
for cla in expression_class:
    mkfile('data/train/'+cla)
    
# 创建 验证集val 文件夹，并由5种类名在其目录下创建5个子目录
mkfile('data/val')
for cla in expression_class:
    mkfile('data/val/'+cla)

mkfile('data/test')
for cla in expression_class:
    mkfile('data/test/'+cla)


split_rate = 0.2


for cla in expression_class:
    cla_path = file_path + '/' + cla + '/'  # 某一个表情的子目录
    images = os.listdir(cla_path)		    # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate)) # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
    	# eval_index 中保存验证集val的图像名称
        if image in eval_index:					
            image_path = cla_path + image
            new_path = 'data/val/' + cla
            copy(image_path, new_path)  # 将选中的图像复制到新路径
           
        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()

val_path = 'data/val'

for cla in expression_class:
    cla_path = val_path + '/' + cla + '/'  # 某一个表情的子目录
    images = os.listdir(cla_path)		    # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    eval_index = random.sample(images, k=int(num*0.5)) # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
    	# eval_index 中保存验证集val的图像名称
        if image in eval_index:					
            image_path = cla_path + image
            new_path = 'data/test/' + cla
            move(image_path, new_path)  # 将选中的图像复制到新路径

print("processing done!")
