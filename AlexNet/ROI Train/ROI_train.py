from torch._C import device
import torch.nn as nn
import torch
from torchvision import models,transforms, datasets
import torch.optim as optim
import time

dict = {"left_eye" : 1, "right_eye" : 2, 
        "nose" : 3, "mouth" : 4, 
        "eyes_nose" : 5, "nose_mouth" : 6, 
        "face" : 7}

def train(ROI):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

        "val": transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


    image_path = "./data/" 				 		

    # 导入训练集并进行预处理
    train_dataset = datasets.ImageFolder(root=image_path + "/train/" + ROI,		
                                        transform=data_transform["train"])
    train_num = len(train_dataset)

    # 按batch_size分批次加载训练集
    train_loader = torch.utils.data.DataLoader(train_dataset,	# 导入的训练集
                                            batch_size=32, 	# 每批训练的样本数
                                            shuffle=True,	# 是否打乱训练集
                                            num_workers=0)	# 使用线程数，在windows下设置为0

    # 导入验证集并进行预处理
    validate_dataset = datasets.ImageFolder(root=image_path + "/val/" + ROI,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)

    # 加载验证集
    validate_loader = torch.utils.data.DataLoader(validate_dataset,	# 导入的验证集
                                                batch_size=32, 
                                                shuffle=True,
                                                num_workers=0)


    #调用alexnet模型，pretrained=True表示读取网络结构和预训练模型，False表示只加载网络结构，不需要预训练模型
    model = models.alexnet(pretrained=True)  #只加载结构
    model.classifier[6] = nn.Linear(4096,8)

    # device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())

    save_path = './ROI Train/' + ROI + '_AlexNet.pth'
    best_acc = 0.0

    for epoch in range(10):
        ########################################## train ###############################################
        model.train()     					# 训练过程中开启 Dropout
        running_loss = 0.0					# 每个 epoch 都会对 running_loss  清零
        time_start = time.perf_counter()	# 对训练一个 epoch 计时
        
        for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
            images, labels = data   # 获取训练集的图像和标签
            optimizer.zero_grad()	# 清除历史梯度
            
            outputs = model(images.to(device))				 # 正向传播
            loss = loss_function(outputs, labels.to(device)) # 计算损失
            loss.backward()								     # 反向传播
            optimizer.step()								 # 优化器更新参数
            running_loss += loss.item()
            
            # 打印训练进度（使训练过程可视化）
            rate = (step + 1) / len(train_loader)           # 当前进度 = 当前step / 训练一轮epoch所需总step
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print('%f s' % (time.perf_counter()-time_start))

        ########################################### validate ###########################################
        model.eval()    # 验证过程中关闭 Dropout
        acc = 0.0  
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                acc += (predict_y == val_labels.to(device)).sum().item()    
            val_accurate = acc / val_num
            
            # 保存准确率最高的那次网络参数
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(), save_path)
                
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
                (epoch + 1, running_loss / step, val_accurate))

    print('Finished Training ' + ROI)
    print('best:', best_acc)


for i in dict.keys():
    train(i)
