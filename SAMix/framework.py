"""
Based on https://github.com/weiyao1996/ScRoadExtractor
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as Func
from cutout import Cutout
from RoadPaste_batch import RoadPaste_batch, MixOutput


class Myframe_SA_MixNet():
    def __init__(self, net, net_D, loss, lr=2e-6, lr_D=5e-6, evalmode=False):
        print('lr:', lr, 'lr_D:', lr_D)
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
        self.loss = loss.cuda()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

        self.net_D = net_D().cuda()
        self.net_D = torch.nn.DataParallel(self.net_D, device_ids=range(torch.cuda.device_count()))
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(0.9, 0.99))
        # self.optimizer_D = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
        self.old_lr_D = lr_D
        if evalmode:
            for i in self.net_D.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

        self.bce_loss = nn.BCEWithLogitsLoss().cuda()
        self.L1_loss = nn.L1Loss()

    def set_input(self, img_batch, mask_batch=None, buffer_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.buffer = buffer_batch
        self.img_id = img_id

    def set_val_input(self, img_batch, img_id=None):
        self.img = img_batch
        self.img_id = img_id

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        if self.buffer is not None:
            self.buffer = V(self.buffer.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        # ------------------------Train origin data----------------------
        self.net.train()
        self.net_D.eval()

        pred_origin = self.net.forward(self.img)

        index_true = (torch.ones(size = self.mask.shape)).cuda()
        index_true[torch.where(self.mask == 128)] = 0

        pred_D = torch.mul(pred_origin, index_true)

        D_out = self.net_D(self.img, pred_D)
        loss_adv = self.bce_loss(D_out, torch.ones_like(D_out))
        loss_adv = loss_adv.cuda()
        loss_adv = loss_adv.type(torch.cuda.FloatTensor)

        # -----------------------Train mix data--------------------------
        # mix
        img_mix, mask_mix, transpose_mask, transpose_mask_fix, label = RoadPaste_batch(self.img, self.buffer, self.mask)
        # cutout
        img_cut, mask_cut, transpose_cut = Cutout(img_mix, mask_mix)
        mask_cut, img_cut = mask_cut.cuda(), img_cut.cuda()
        pred_cut = self.net.forward(img_cut)

        # cutout_loss
        loss_cut = self.loss(pred_cut, mask_cut)
        loss_cut = loss_cut.cuda()
        loss_cut = loss_cut.type(torch.cuda.FloatTensor)

        ### mixed output
        mixed_pred = MixOutput(pred_origin, transpose_mask, transpose_mask_fix, transpose_cut, label)

        loss_origin = self.loss(pred_origin, self.mask)
        loss_origin = loss_origin.cuda()
        loss_origin = loss_origin.type(torch.cuda.FloatTensor)

        # loss_inv = 1 - Func.cosine_similarity(mixed_pred.detach(), pred_cut, dim=1).mean()
        loss_inv = 1 - Func.cosine_similarity(mixed_pred, pred_cut, dim=1).mean()
        loss_inv = loss_inv.cuda()
        loss_inv = loss_inv.type(torch.cuda.FloatTensor)

        loss_final = loss_origin + loss_cut + loss_inv * 0.1 + loss_adv * 0.1  # + loss_invariant
        loss_final.backward()
        self.optimizer.step()

        #-------------------------Train Discriminator-------------------
        # Train with pred
        self.net_D.train()
        pred1 = pred_D.detach()
        # pred = torch.mul(pred, index_true)
        D_out1 = self.net_D(self.img, pred1)
        loss_D_fake = self.bce_loss(D_out1, torch.zeros_like(D_out1))
        loss_D_fake = loss_D_fake.cuda()
        loss_D_fake = loss_D_fake.type(torch.cuda.FloatTensor)

        # Train with label
        D_out2 = self.net_D(self.img, torch.mul(self.mask, index_true))
        loss_D_true = self.bce_loss(D_out2, torch.ones_like(D_out2))
        loss_D_true = loss_D_true.cuda()
        loss_D_true = loss_D_true.type(torch.cuda.FloatTensor)

        loss_D = loss_D_true + loss_D_fake
        loss_D.backward()
        self.optimizer_D.step()

        return loss_adv.item(), loss_origin.item(), loss_cut.item(), loss_inv.item(), loss_final.item()

    def validation(self):
        self.img = self.img.cuda()
        self.net.eval()
        with torch.no_grad():
            output = self.net(self.img)
        return output

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def save_d(self, path):
        torch.save(self.net_D.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def load_d(self, path):
        self.net_D.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def update_lr_d(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr_D / new_lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate_D: %f -> %f' % (self.old_lr_D, new_lr), file=mylog)
        print('update learning rate_D: %f -> %f' % (self.old_lr_D, new_lr))
        self.old_lr_D = new_lr