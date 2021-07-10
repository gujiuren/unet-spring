import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
from utils.tools import add_samples

train_data_path = r"M:\DRIVE\training"
eval_data_path = r"M:\DRIVE\test"

trans = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, ], [0.5, ])
])

class UnetDataset(Dataset):
    def __init__(self, data_path, is_train = True):
        super(UnetDataset, self).__init__()
        self.is_train = is_train

        img_path = os.path.join(data_path, "images")
        label_path = os.path.join(data_path, "1st_manual")

        self.images_path = [img_path + "/" + image_name for image_name in os.listdir(img_path)]
        self.labels_path = [label_path + "/" + label_name for label_name in os.listdir(label_path)]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        # 生成抠图的随机坐标
        random_x = np.random.randint(0, 565 - 97)
        random_y = np.random.randint(0, 584 - 97)

        image = cv2.imread(self.images_path[index], 1)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        label = Image.open(self.labels_path[index])
        label = cv2.cvtColor(np.asarray(label), cv2.COLOR_GRAY2BGR)
        label = Image.fromarray(cv2.cvtColor(label, cv2.COLOR_BGR2RGB))

        image_crop, label_crop = add_samples(image, label)
        # image_crop = image.crop((random_x, random_y, random_x+96, random_y+96))
        # label_crop = label.crop((random_x, random_y, random_x+96, random_y+96))
        if self.is_train == True:
            image_data = trans(image_crop)
            label_data = trans(label_crop)
        else:
            image_data = trans(image)
            label_data = trans(label)

        return image_data, label_data



if __name__ == '__main__':
    dataset = UnetDataset(train_data_path)
    print(len(dataset))
    image_crop, label_crop = dataset[0]
    print(image_crop)
    print("="*200)
    print(label_crop)
