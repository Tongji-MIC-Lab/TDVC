import random

import albumentations as A
import torch
import torchvision


def imgauglist(input_image, ref_imagelist, size):
    transform = A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.OneOf([A.RGBShift(p=0.5), A.RandomBrightnessContrast(p=0.5)], p=0.5),
        A.RandomSizedCrop([size[0], size[0]], size[0], size[1], p=1)])

    data = transform(image=input_image)

    images = []
    input_image = A.ReplayCompose.replay(data['replay'], image=input_image)['image']
    images.append(torchvision.transforms.ToTensor()(input_image))

    for x in ref_imagelist:
        ref_image = A.ReplayCompose.replay(data['replay'], image=x)['image']
        images.append(torchvision.transforms.ToTensor()(ref_image))

    images = torch.stack(images)
    return images[0], images[1:]


def imgauglist2(input_image, ref_imagelist, size):
    transform = A.ReplayCompose([
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.OneOf([
        #     A.RGBShift(p=0.5),
        #     A.RandomBrightnessContrast(p=0.5),
        # ], p=0.5),
        # A.OneOf([
        #     A.RandomSizedCrop([min(input_image.shape[:2]), min(input_image.shape[:2])], size[0], size[1], p=1),
        # ], p=1)

        ###############################
        ###############################
        ###############################

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.OneOf([A.RGBShift(p=0.5), A.RandomBrightnessContrast(p=0.5)], p=0.5)])

    data = transform(image=input_image)

    transform2 = A.ReplayCompose([A.RandomSizedCrop([size[0], size[0]], size[0], size[1], p=1)])
    data2 = transform2(image=input_image)

    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.RandomResizedCrop((size[0], size[1]), scale=(0.5, 1.0))])

    resize = False
    if random.random() < 0.5:
        resize = True

    images = []
    input_image = A.ReplayCompose.replay(data['replay'], image=input_image)['image']
    if resize:
        input_image = A.ReplayCompose.replay(data2['replay'], image=input_image)['image']
    images.append(torchvision.transforms.ToTensor()(input_image))

    for x in ref_imagelist:
        ref_image = A.ReplayCompose.replay(data['replay'], image=x)['image']
        if resize:
            ref_image = A.ReplayCompose.replay(data2['replay'], image=ref_image)['image']
        images.append(torchvision.transforms.ToTensor()(ref_image))

    images = torch.stack(images)
    if not resize:
        images = transforms(images)

    return images[0], images[1:]
