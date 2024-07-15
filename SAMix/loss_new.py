import torch
import torch.nn as nn

class partial_BCELoss(nn.Module):
    def __init__(self, a=1.0):
        super(partial_BCELoss, self).__init__()
        self.a = a

    def forward(self, y_pred, y_true):
        index_true = (torch.zeros(size=y_true.shape)).cuda()
        index_true[y_true != 128] = 1

        weight = (torch.zeros(size=y_true.shape)).cuda()
        weight[y_true == 255] = self.a
        weight[y_true == 0] = 1
        # print(weight)

        y_true = y_true / 255.0
        y_true[y_true > 0.5] = 1
        y_true[y_true <= 0.5] = 0
        smooth = 0.00001
        bce_loss = -(y_true * torch.log(y_pred + smooth) + (1 - y_true) * torch.log(1 - y_pred + smooth))
        pCEloss = (torch.sum(torch.mul(torch.mul(bce_loss, index_true), weight)) + smooth) / (
                    torch.sum(index_true) + smooth)

        return pCEloss

class PBCELoss(nn.Module):
    def __init__(self, a=1.0):
        super(PBCELoss, self).__init__()
        self.a = a

    def forward(self, y_pred, y_true):
        index_true = (torch.ones(size=y_true.shape)).cuda()
        index_true[y_true == 128] = 0

        index_true[y_true == 255] = self.a

        y_true = y_true / 255.0
        y_true[y_true > 0.5] = 1
        y_true[y_true <= 0.5] = 0
        smooth = 0.00001
        # print(torch.where(y_true == 1))
        bce_loss = -(y_true * torch.log(y_pred + smooth) + (1 - y_true) * torch.log(1 - y_pred + smooth))
        pCEloss = (torch.sum(torch.mul(bce_loss, index_true)) + smooth) / (torch.sum(index_true) + smooth)
        # print(pCEloss)
        return pCEloss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.float() / 255.0  # Assuming y_true is initially uint8 with values in {0, 128, 255}
        y_true[y_true > 0.5] = 1.0  # Convert to binary {0, 1}
        y_true[y_true <= 0.5] = 0.0  # values

        smooth = 1e-5
        bce_loss = -(y_true * torch.log(y_pred + smooth) + (1 - y_true) * torch.log(1 - y_pred + smooth))
        return torch.mean(bce_loss)  # The mean loss over all elements in the batch