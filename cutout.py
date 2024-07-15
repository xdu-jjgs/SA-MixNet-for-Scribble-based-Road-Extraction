import torch
import numpy as np
import PIL
import torchvision.transforms.functional as F
import torch
from torch.autograd import Variable as V


def Cutout(imgs, labels, n_holes=1, length=128):
    h = imgs.shape[2]
    w = imgs.shape[3]
    num = imgs.shape[0]
    labels_list = []
    imgs_list = []
    masks_list = []

    for i in range(num):
        label = labels[i, :, :, :]
        img = imgs[i, :, :, :]
        mask = np.ones((1, h, w), np.float32)
        mask = torch.from_numpy(mask)


        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[0, y1: y2, x1: x2] = 0.

        # mask = mask.expand_as(img)
        # print(np.shape(mask))
        # print(np.shape(img))
        # print(img, mask)
        mask = V(mask.cuda(), volatile=False)
        img = img * mask
        label = label * mask

        imgs_list.append(img)
        labels_list.append(label)
        masks_list.append(mask)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    masks_out = torch.stack(masks_list)

    return imgs_out, labels_out, masks_out

def Cutout_withedge(imgs, labels, heds, n_holes=1, length=128):
    h = imgs.shape[2]
    w = imgs.shape[3]
    num = imgs.shape[0]
    labels_list = []
    imgs_list = []
    heds_list = []
    masks_list = []

    for i in range(num):
        label = labels[i, :, :, :]
        img = imgs[i, :, :, :]
        hed = heds[i, :, :, :]
        mask = np.ones((1, h, w), np.float32)
        mask = torch.from_numpy(mask)


        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[0, y1: y2, x1: x2] = 0.

        # mask = mask.expand_as(img)
        # print(np.shape(mask))
        # print(np.shape(img))
        # print(img, mask)
        mask = V(mask.cuda(), volatile=False)
        img = img * mask
        label = label * mask
        hed = hed * mask

        imgs_list.append(img)
        labels_list.append(label)
        heds_list.append(hed)
        masks_list.append(mask)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    heds_out = torch.stack(heds_list)
    masks_out = torch.stack(masks_list)

    return imgs_out, labels_out, heds_out, masks_out

def rotate_invariant(imgs, labels):
    num = imgs.shape[0]
    imgs_out_list = []
    labels_out_list = []
    angles = []

    for i in range(num):
        img = imgs[i, :, :, :]
        label = labels[i, :, :, :]

        angle = float(torch.empty(1).uniform_(0.0, 360.0).item())

        rotated_img = F.rotate(img, angle, PIL.Image.NEAREST, False, None)
        rotated_label = F.rotate(label, angle, PIL.Image.NEAREST, False, None)

        imgs_out_list.append(rotated_img)
        labels_out_list.append(rotated_label)

        angles.append(angle)

    imgs_out = torch.stack(imgs_out_list)
    labels_out = torch.stack(labels_out_list)
    return imgs_out, labels_out, angles


def rotate_back(imgs, outputs, labels, angles):
    num = imgs.shape[0]
    imgs_out_list = []
    outputs_out_list = []
    labels_out_list = []

    for i in range(num):
        img = imgs[i, :, :, :]
        output = outputs[i, :, :, :]
        label = labels[i, :, :, :]
        angle = -angles[i]

        rotated_img = F.rotate(img, angle, PIL.Image.NEAREST, False, None)
        rotated_output = F.rotate(output, angle, PIL.Image.NEAREST, False, None)
        rotated_label = F.rotate(label, angle, PIL.Image.NEAREST, False, None)

        imgs_out_list.append(rotated_img)
        outputs_out_list.append(rotated_output)
        labels_out_list.append(rotated_label)

    imgs_out = torch.stack(imgs_out_list)
    outputs_out = torch.stack(outputs_out_list)
    labels_out = torch.stack(labels_out_list)
    return imgs_out, outputs_out, labels_out