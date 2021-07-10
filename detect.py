import os
import cv2
import torch
import numpy as np
from PIL import Image
from nets.UNet import MainNet
import torchvision.transforms as transforms

trans = transforms.Compose([
    transforms.ToTensor()
])

eval_img_path = r"M:\DRIVE\test\images"
eval_label_path = r"M:\DRIVE\test\2nd_manual_better"
weights_path = r"F:\mastCloud\Pycharm_work_space\U_Net_v1.1\weights\unet.pt"


def get_img_path(eval_img_path, eval_label_path):
    # 取出所有样本图片路径
    images_name = os.listdir(eval_img_path)
    images_full_path = [eval_img_path + "/" + image_name for image_name in images_name]
    # 取出所有标签图片路径
    labels_name = os.listdir(eval_label_path)
    labels_full_path = [eval_label_path + "/" + label_name for label_name in labels_name]

    return images_full_path, labels_full_path


def detect():
    # 实例化网络
    net = MainNet()
    # 下载权重
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path))
        print("权重已下载")
    # 获得验证路径列表
    images_full_path, labels_full_path = get_img_path(eval_img_path, eval_label_path)
    # 开始验证
    for index, image_full_path in enumerate(images_full_path):
        # 创建一个黑色图片
        black_array = np.zeros(shape=(560, 1680, 3), dtype=np.uint8)
        black_ground = Image.fromarray(black_array)
        # 读取tif图片（输入图片）
        img = cv2.cvtColor(cv2.imread(image_full_path), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((560, 560))
        # 读取gif文件（输入标签）
        label = Image.open(labels_full_path[index]).resize((560, 560))
        # 将输入图片和标签黏贴到黑色图片上
        black_ground.paste(img, (0, 0))
        black_ground.paste(label, (560, 0))
        # 在输入图片上加一个批次
        img = trans(img).unsqueeze(0)
        # 前向和前向处理
        predict = net(img)
        predict = predict.detach().squeeze(0).permute(1, 2, 0)
        predict_img = ((predict * 255).numpy().astype(np.uint8))
        print(predict_img.shape)
        predict_img = Image.fromarray(predict_img)
        black_ground.paste(predict_img, (1120, 0))
        black_ground.save(f"M:\DRIVE\predicts\img_{index}.jpg")
        print("image saved!")


if __name__ == '__main__':
    detect()
