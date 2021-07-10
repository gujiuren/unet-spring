import os
import time
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入网络
from nets.UNet import MainNet
# 导入dataset类
from make_dataset.unet_dataset import UnetDataset

# 训练数据路径和验证数据路径
train_data_path = r"M:\DRIVE\training"
eval_data_path = r"M:\DRIVE\test"
weights_path = r"F:\mastCloud\Pycharm_work_space\U_Net_v1.1\weights\unet.pt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def train_eval_process(batch_size=20, epochs=100):
    # 实例化网络
    net = MainNet().to(device)

    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path))
        print("下载权重完成！")

    # 实例化trian_data和eval_data
    train_dataset = UnetDataset(train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    eval_dataset = UnetDataset(eval_data_path)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,drop_last=True)

    # 实例化优化器和损失函数
    optimizer = optim.Adam(net.parameters())
    loss_func = nn.BCELoss()

    train_avg_losslist = [100]

    for epoch in range(epochs):

        # 训练过程
        train_start_time = time.time()
        train_total_loss = 0
        train_total_accuracy = 0
        net.train()
        for index, (image_data, label_data) in enumerate(train_dataloader):
            image_data, label_data = image_data.to(device), label_data.to(device)

            train_predict = net(image_data)
            loss = loss_func(train_predict, label_data)
            # 三部曲
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            b_predict = (train_predict.cpu() > 0.5).int()
            # print(((torch.eq(b_predict, label_data.cpu().int())).int()).shape)
            train_total_accuracy += torch.sum((torch.eq(b_predict, label_data.cpu())).int()) / (
                    96 * 96 * 3 * batch_size)
            # 求loss
            train_total_loss += loss.detach().item()
        train_avg_accuracy = train_total_accuracy / len(train_dataloader)
        train_avg_loss = train_total_loss / len(train_dataloader)
        if min(train_avg_losslist) > train_avg_loss:
            torch.save(net.state_dict(), weights_path,
                       _use_new_zipfile_serialization=False)
            print("保存权重！")
        train_avg_losslist.append(train_avg_loss)
        print(
            f"epoch:{epoch} loss:{train_avg_loss} accuracy:{train_avg_accuracy} "
            f"time:{time.time() - train_start_time}")

        # 验证过程
        with torch.no_grad():
            eval_start_time = time.time()
            eval_total_loss = 0
            eval_total_accuracy = 0
            net.eval()
            for index, (eval_data, eval_label) in enumerate(eval_dataloader):
                eval_data, eval_label = eval_data.to(device), eval_label.to(device)

                eval_predict = net(eval_data)


                eval_loss = loss_func(eval_predict, eval_label)
                eval_total_loss += eval_loss.detach().item()

                binary_predict = (eval_predict.cpu() > 0.5).int()
                eval_total_accuracy += torch.sum(torch.eq(binary_predict, eval_label.cpu()).int()) / (
                        96 * 96 * 3 * batch_size)


        eval_avg_loss = eval_total_loss / len(eval_dataloader)
        eval_avg_accuracy = eval_total_accuracy / len(eval_dataloader)
        print(
            f"epoch:{epoch} loss:{eval_avg_loss} accuracy:{eval_avg_accuracy} "
            f"time:{time.time() - eval_start_time}")


if __name__ == '__main__':
    train_eval_process(batch_size=20, epochs=10000)
