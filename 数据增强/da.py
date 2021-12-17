import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import random
plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open('flower.jpg')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()
def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    return output_data
def da_by_strategy_with_tensor(image_tensor,strategy):
    return T.ToTensor()(da_strategies[strategy](T.ToPILImage()(image_tensor)))
def da_by_strategy(image,strategy):
    return da_strategies[strategy](image)
def padding(image):
    return T.Pad(padding=random.randint(1,50))(image)
def resize(image):
    return T.Resize(size=random.randint(50,100))(image)
def center_crop(image):
    return T.CenterCrop(size=random.randint(0,image.size[0]))(orig_img)
def five_crop(orig_img):
    return orig_img
def gray_scale(orig_img):
    return T.Grayscale()(orig_img)
def color_jitter(orig_img):
    return T.ColorJitter(brightness=.5, hue=.3)(orig_img)
def gaussian_blur(orig_img):
    return T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(orig_img)
def random_prespective(orig_img):
    return T.RandomPerspective(distortion_scale=0.6, p=1.0)(orig_img)
def random_rotation(orig_img):
    return T.RandomRotation(degrees=(0, 180))(orig_img)
def random_affine(orig_img):
    return T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))(orig_img)
def random_crop(orig_img):
    return T.RandomCrop(size=(128, 128))(orig_img)
def random_resized_crop(orig_img):
    return T.RandomResizedCrop(size=(32, 32))(orig_img)
def random_invert(orig_img):
    return T.RandomInvert()(orig_img)
def random_posterize(orig_img):
    return T.RandomPosterize(bits=2)(orig_img)
def random_solarize(orig_img):
    return T.RandomSolarize(threshold=192.0)(orig_img)
def random_adjust_sharpness(orig_img):
    return T.RandomAdjustSharpness(sharpness_factor=2)(orig_img)
def random_autocontrast(orig_img):
    return T.RandomAutocontrast()(orig_img)
def random_equalize(orig_img):
    return T.RandomEqualize()(orig_img)
def auto_augment(orig_img):
    return T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)(orig_img)
def rand_augment(orig_img):
    return T.RandAugment()(orig_img)
def trivial_augment_wide(orig_img):
    return T.TrivialAugmentWide()(orig_img)
def random_horizontal_flip(orig_img):
    return T.RandomHorizontalFlip(p=0.5)(orig_img)
def random_vertical_flip(orig_img):
    return T.RandomVerticalFlip(p=0.5)(orig_img)
def none_da(orig_img):
    return orig_img
torch.manual_seed(0)
# none_da, padding, resize, center_crop, five_crop, gray_scale, color_jitter, gaussian_blur,
#                  random_prespective, random_rotation, random_affine, random_crop, random_resized_crop,
#                  random_invert, random_posterize, random_solarize, random_adjust_sharpness, random_autocontrast,
#                  random_equalize, auto_augment,random_augment,trivial_augment_wide
da_strategies = [none_da,five_crop,random_resized_crop,gaussian_blur,gray_scale]
if __name__=='__main__':
    image=Image.open('flower.jpg')
    image_tensor=T.ToTensor()(image)
    print(image_tensor.shape)
    image=da_by_strategy_with_tensor(image_tensor,0)
    print(image_tensor.shape)
    plt.imshow(T.ToPILImage()(image))
    plt.show()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # for i in range(len(da_strategies)):
    #
    #     plt.imshow(da_strategies[i](image))
    #     plt.show()