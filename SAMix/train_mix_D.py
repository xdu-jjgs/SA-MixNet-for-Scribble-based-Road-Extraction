"""
Based on https://github.com/weiyao1996/ScRoadExtractor
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch

import torch.utils.data as data

from tqdm import tqdm
from data import ImageFolder, ImageFolder_test

from networks.discriminator import *
from framework import *
from loss_new import PBCELoss, partial_BCELoss
from loss import *
import numpy as np
from networks.dinknet import DinkNet34_new
from tensorboardX import SummaryWriter

data = 'WHU'
data1 = 'WHU'
SHAPE = (512, 512)

sat_dir = './data/' + data + '/train/sat/'
lab_dir = './data/' + data1 + '/train/proposal_mask/'
buffer_dir = './data/' + data1 + '/train/buffer_mask/'
imagelist = os.listdir(buffer_dir)
trainlist = map(lambda x: x[:-9], imagelist)

test_sat_dir = './data/' + data + '/val/sat/'
test_lab_dir = './data/' + data + '/val/mask/'
test_imagelist = os.listdir(test_lab_dir)
testlist = map(lambda x: x[:-9], test_imagelist)

NAME = 'SA_MixNet_WHU_date[x.x]'
BATCHSIZE_PER_CARD = 2

solver = Myframe_SA_MixNet(DinkNet34_new, Discriminator, PBCELoss(2), 2e-4, 2e-4)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, sat_dir, lab_dir, buffer_dir)
test_dataset = ImageFolder_test(testlist, test_sat_dir, test_lab_dir)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0,
    drop_last=True)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=0,
    drop_last=True
)

mylog = open('./logs/' + NAME + '.log', 'w')
logger = SummaryWriter(log_dir='./logs//' + NAME + '_tensorboardX')
tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.

best_iou = 0.0

for epoch in range(0, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_loss_adv = 0.0
    train_loss_origin = 0.0
    train_loss_cut = 0.0
    train_loss_inv = 0.0
    train_loss_final = 0.0
    for img, mask, hed, buffer in tqdm(data_loader_iter):
        solver.set_input(img, mask, buffer)
        loss_adv, loss_origin, loss_cut, loss_inv, loss_final = solver.optimize()
        train_loss_adv += loss_adv
        train_loss_origin += loss_origin
        train_loss_cut += loss_cut
        train_loss_inv += loss_inv
        train_loss_final += loss_final
    train_loss_adv /= len(data_loader_iter)
    train_loss_origin /= len(data_loader_iter)
    train_loss_cut /= len(data_loader_iter)
    train_loss_inv /= len(data_loader_iter)
    train_loss_final /= len(data_loader_iter)
    logger.add_scalar("train loss adv", train_loss_adv, global_step=epoch)
    logger.add_scalar("train loss origin", train_loss_origin, global_step=epoch)
    logger.add_scalar("train loss cut", train_loss_cut, global_step=epoch)
    logger.add_scalar("train loss inv", train_loss_inv, global_step=epoch)
    logger.add_scalar("train loss final", train_loss_final, global_step=epoch)

    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('train_loss_adv:', train_loss_adv, file=mylog)
    print('train_loss_origin:', train_loss_origin, file=mylog)
    print('train_loss_cut:', train_loss_cut, file=mylog)
    print('train_loss_inv:', train_loss_inv, file=mylog)
    print('train_loss_final', train_loss_final, file=mylog)
    print('SHAPE:', SHAPE, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    # print('train_loss:', train_epoch_loss)
    print('SHAPE:', SHAPE)

    test_data_loader_iter = iter(test_data_loader)
    matrix = np.zeros((2, 2), dtype=np.int64)
    for img, mask in tqdm(test_data_loader_iter):
        solver.set_val_input(img)
        output = solver.validation()
        pred = output.data.cpu().numpy()
        mask1 = mask.data.cpu().numpy()
        mask1 /= 255
        pred[np.where(pred > 0.5)] = 1
        pred[np.where(pred < 0.5)] = 0
        count = np.bincount(2 * mask.reshape(-1) + pred.reshape(-1), minlength=2 ** 2)
        matrix += count.reshape((2, 2))
    precisions = np.diag(matrix) / matrix.sum(axis=0)
    print('precisions: ', precisions)
    print('precisions: ', precisions, file=mylog)
    recalls = np.diag(matrix) / matrix.sum(axis=1)
    print('recalls: ', recalls)
    print('recalls: ', recalls, file=mylog)
    f1s = 2 * precisions * recalls / (precisions + recalls)
    print('f1s: ', f1s)
    print('f1s: ', f1s, file=mylog)
    intersection = np.diag(matrix)
    union = np.sum(matrix, axis=0) + np.sum(matrix, axis=1) - np.diag(matrix)
    ious = intersection / union
    print('ious: ', ious, file=mylog)
    print('ious: ', ious)
    logger.add_scalar("precisions", precisions[1], global_step=epoch)
    logger.add_scalar("recalls", recalls[1], global_step=epoch)
    logger.add_scalar("f1s", f1s[1], global_step=epoch)
    logger.add_scalar("ious", ious[1], global_step=epoch)
    if ious[1] > best_iou:
        best_iou = ious[1]
        solver.save('./weight/' + NAME + '_' + str(epoch) + '_iou_' + str(best_iou) + '.th')
        solver.save_d('./weight/' + NAME + '_' + str(epoch) + '_iou_' + str(best_iou) + '_d.th')

    if train_loss_final >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_loss_final
        solver.save('./weight/' + NAME + '.th')
        solver.save_d('./weight/' + NAME + '_d.th')
    if no_optim > 12:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 6:
        if solver.old_lr < 5e-9:
            break
        solver.load('./weight/' + NAME + '.th')
        solver.load_d('./weight/' + NAME + '_d.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
        solver.update_lr_d(5.0, factor=True, mylog=mylog)
        no_optim = 0
    mylog.flush()


print('Finish!', file=mylog)
print('Finish!')
mylog.close()
