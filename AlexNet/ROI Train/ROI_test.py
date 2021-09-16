import os
import torch
from PIL import Image
from torchvision import transforms,models
import matplotlib.pyplot as plt
import numpy as np

# 预处理
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dict = {"left_eye" : 1, "right_eye" : 2, 
        "nose" : 3, "mouth" : 4, 
        "eyes_nose" : 5, "nose_mouth" : 6, 
        "face" : 7}

#class_indict
class_indict = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sadness', 7: 'surprise'}

def origin_test(ROI):
    # create model
    model = models.alexnet(pretrained=False)  #只加载结构
    model.classifier[6] = torch.nn.Linear(4096,8)

    # load model weights
    model_weight_path = ROI + "_AlexNet.pth"
    model.load_state_dict(torch.load(model_weight_path))

    # 关闭 Dropout
    model.eval()

    sum = 0
    error = 0
    for root, dirs, files in os.walk('..\\data\\test\\' + ROI):

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
            # print(predict)
            print(predict_class, predict_possibility)
        print()
    print('Accuracy : %.2f' %((1 - error/sum)*100),'%')
    print("----------------------------------------------")

# for i in dict.keys():
#     test(i)

def test(img, ROI):

    # create model
    model = models.alexnet(pretrained=False)  #只加载结构
    model.classifier[6] = torch.nn.Linear(4096,8)
    # load model weights
    model_weight_path = "./ROI Train/" + ROI + "_AlexNet.pth"
    model.load_state_dict(torch.load(model_weight_path))

        # 关闭 Dropout
    model.eval()

    # load image
    img = Image.open(img)

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

    return predict_class, predict_possibility, predict

def majority_voting():
    for i in class_indict.keys():
        res = []
        for ROI in dict.keys():     
            for root, dirs, files in os.walk('.\\data\\test\\' + ROI + '\\' + class_indict[i]):
                row = []
                for name in files:
                    img_path = os.path.join(root, name)
                    predict_class, predict_possibility, predict = test(img_path, ROI)
                    row.append(predict_class)
                res.append(row)

        res = list(map(list, zip(*res)))
        
        r = [max(j, key=j.count) for j in res]

        print(class_indict[i] + " : ", r.count(class_indict[i])/len(r))


def median_simple():
    for i in class_indict.keys():
        res = []
        for ROI in dict.keys():     
            for root, dirs, files in os.walk('.\\data\\test\\' + ROI + '\\' + class_indict[i]):
                row = []
                for name in files:
                    img_path = os.path.join(root, name)
                    predict_class, predict_possibility, predict = test(img_path, ROI)
                   
                    row.append(predict.tolist())
                res.append(row)

        
        r = np.array(res)
        
        # r = np.median(r, axis = 0) #median rule
        r = r.mean(axis = 0) #simple average rule

        r = np.argmax(r, axis = 1)

        print(class_indict[i] + " : ", len(np.where(r==i)[0])/len(r))

# median_simple()
majority_voting()