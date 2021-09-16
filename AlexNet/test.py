import os
import torch
from PIL import Image
from torchvision import transforms,models
import matplotlib.pyplot as plt

# 预处理
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#class_indict
class_indict = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sadness', 7: 'surprise'}

# create model
model = models.alexnet(pretrained=False)  #只加载结构
model.classifier[6] = torch.nn.Linear(4096,8)

# load model weights
model_weight_path = "AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))

# 关闭 Dropout
model.eval()

sum = 0
error = 0
for root, dirs, files in os.walk('.\\data\\test'):

    test_class = root.split('\\')[-1]
    print(test_class, ':')

    for name in files:
        img_path = os.path.join(root, name)

        # load image
        img = Image.open(img_path)

        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img))     # 将输出压缩，即压缩掉 batch 这个维度
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        
        predict_class = class_indict[int(str(predict_cla))]
        predict_possibility = predict[predict_cla].item()

        if(test_class != predict_class):
            error += 1
        sum += 1
        print(predict)
        print(predict_class, predict_possibility)
        break

print('Accuracy : %.2f' %((1 - error/sum)*100),'%')