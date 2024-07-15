"""
Based on https://github.com/weiyao1996/ScRoadExtractor
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
from tqdm import tqdm

import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np

from time import time
from networks.dinknet import *
from networks.dinknet import DinkNet34_new

BATCHSIZE_PER_CARD = 2

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())

        # maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        # maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        # maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        # maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        maska = self.net.forward(img1)
        maskb = self.net.forward(img2)
        maskc = self.net.forward(img3)
        maskd = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()


        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        # print()
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


class TTAFrame_one():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = np.array(img)
        img = img.transpose(2, 0, 1)
        # 为img添加一个维度变为1*3*512*512
        img = np.expand_dims(img, axis=0)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        mask = self.net.forward(img).squeeze().cpu().data.numpy()

        return mask

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

# testdir = '/home/ck/data/wuhan/train/sat/'
testdir = r'C:\Users\HuangHao\Desktop\mini_MA/'
test = os.listdir(testdir)
solver = TTAFrame_one(DinkNet34_new)
solver.load('weights/epoch_100.pth')
tic = time()
# target_grey = '/home/ck/data/wuhan/train_new_losses/pred_dlinknet88_edge33/'
target_grey = r'./test/'
os.makedirs(target_grey, exist_ok=True)

for name in tqdm(test): # !!!!!!
    mask = solver.test_one_img_from_path(testdir + name)
    print(np.shape(mask))
    # 创建热力图并保存
    mask = (mask * 255).astype('uint8')  # 确保mask被转换为0-255的uint8类型
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    cv2.imwrite(target_grey + name[:-7] + "_heatmap.png", heatmap)

    # generate gray predicted result
    # mask[mask < 0] = 0
    # mask = (mask / 8) * 255
    #
    # edge[edge < 0] = 0
    # edge = (edge / 8) * 255

    # cv2.imwrite(target_grey+name[:-7]+"pred.png", mask.astype(np.uint8))

    # cv2.imwrite(target_grey+name[:-7]+"edge.png", edge.astype(np.uint8))
