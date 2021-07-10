import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

trans = transforms.Compose([
    transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(hue=0.5)
])


# PIL导入图片,适用于U-Net和U2-Net
def add_samples(pil_train_img, pil_label_img, output_size=96, is_crop=True, is_flip=True, is_color_jitter=True):
    train_img = pil_train_img
    label_img = pil_label_img

    # 获得图片尺寸
    w, h = train_img.size

    # 随机裁剪
    if is_crop == True:
        crop_size = np.random.randint(int(output_size * 0.7), int(output_size * 1.3))
        random_w = np.random.randint(0, (w - crop_size))
        random_h = np.random.randint(0, (h - crop_size))

        train_img = train_img.crop((random_w, random_h, random_w + crop_size, random_h + crop_size))
        label_img = label_img.crop((random_w, random_h, random_w + crop_size, random_h + crop_size))

    # 随机翻转
    if is_flip == True:
        # 随机水平翻转
        random_horizontal = np.random.randint(0, 2)
        if random_horizontal == 1:
            train_img = train_img.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机垂直翻转
        random_vertical = np.random.randint(0, 2)
        if random_vertical == 1:
            train_img = train_img.transpose(Image.FLIP_TOP_BOTTOM)
            label_img = label_img.transpose(Image.FLIP_TOP_BOTTOM)

    # 随机亮度、对比度、色度
    if is_color_jitter == True:
        train_img = trans(train_img)
        label_img = label_img

    # TODO:随机旋转
    # random_angle = np.random.randint(-90, 90)
    #
    # train_img = train_img.rotate(random_angle)
    # label_img = label_img.rotate(random_angle)
    train_img = train_img.resize((output_size, output_size))
    label_img = label_img.resize((output_size, output_size))
    return train_img, label_img


if __name__ == '__main__':
    train_img = Image.open('M:/DRIVE/eyes.jpg')
    label_img = Image.open('M:/DRIVE/eyes.jpg')
    for i in range(5):
        train_result, label_result = add_samples(train_img, label_img)
        train_result.show()
        label_result.show()
