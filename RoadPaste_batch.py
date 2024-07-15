import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch.autograd import Variable as V
import numpy as np
import cv2
import os

import torch
from time import time

from cutout import Cutout, rotate_back, rotate_invariant

def calculate_histogram(tensor_image, bins=25):
    tensor_image = tensor_image.cpu()
    image = np.array(((tensor_image + 1.6) * (255.0 / 3.2)).clamp(0, 255).byte().permute(1, 2, 0))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def compare_histograms(hist1, hist2):
    return cv2.compareHist(np.array(hist1.cpu()), np.array(hist2.cpu()), cv2.HISTCMP_CORREL)

def are_tensors_similar(tensor1, tensor2, threshold=0.5):
    hist1 = calculate_histogram(tensor1)
    hist2 = calculate_histogram(tensor2)
    similarity_score = compare_histograms(hist1, hist2)
    # print('simi:', similarity_score)
    # return similarity_score > threshold
    return similarity_score

def RoadPaste_batch(samples, targets_buffer, targets_mask):
    batch = samples.shape[0]
    if batch == 2:
        samples_mix, targets_mix, transport_mask, transport_fix_mask, label= RoadPaste_batch_2(samples, targets_buffer, targets_mask)
    elif batch == 4:
        samples_mix, targets_mix, transport_mask, transport_fix_mask, label = RoadPaste_batch_4(samples, targets_buffer, targets_mask)
    elif batch == 8:
        samples_mix, targets_mix, transport_mask, transport_fix_mask, label = RoadPaste_batch_8(samples, targets_buffer, targets_mask)
    elif batch == 16:
        samples_mix, targets_mix, transport_mask, transport_fix_mask, label = RoadPaste_batch_16(samples, targets_buffer, targets_mask)
    elif batch == 32:
        samples_mix, targets_mix, transport_mask, transport_fix_mask, label = RoadPaste_batch_32(samples, targets_buffer, targets_mask)

    return samples_mix, targets_mix, transport_mask, transport_fix_mask, label

def MixOutput(pred, transpose_mask, transpose_mask_fix, transpose_cut, label):
    batch = pred.shape[0]
    if batch == 2:
        MixedOutput = MixOutput2(pred, transpose_mask, transpose_mask_fix, transpose_cut, label)
    elif batch == 4:
        MixedOutput = MixOutput4(pred, transpose_mask, transpose_mask_fix, transpose_cut, label)
    elif batch == 8:
        MixedOutput = MixOutput8(pred, transpose_mask, transpose_mask_fix, transpose_cut, label)
    elif batch == 16:
        MixedOutput = MixOutput16(pred, transpose_mask, transpose_mask_fix, transpose_cut, label)
    elif batch == 32:
        MixedOutput = MixOutput32(pred, transpose_mask, transpose_mask_fix, transpose_cut, label)

    return MixedOutput

def MixOutput2(pred, transpose_mask, transpose_mask_fix, transpose_cut, label):
    mixed_out_list = []
    pred1 = pred[0, :, :, :]
    pred2 = pred[1, :, :, :]
    label_ = label[0, 0]

    transpose_mask12 = transpose_mask[0, :, :, :]
    transpose_mask21 = transpose_mask[1, :, :, :]

    transpose_mask_fix12 = transpose_mask_fix[0, :, :, :]
    transpose_mask_fix21 = transpose_mask_fix[1, :, :, :]

    transpose_cut1 = transpose_cut[0, :, :, :]
    transpose_cut2 = transpose_cut[1, :, :, :]
    # print(torch.max(transpose_mask12))

    if label_ == 1:
        output1 = (pred2 * transpose_mask12 + pred1 * (1 - transpose_mask12)) * transpose_cut1
        output2 = (pred1 * transpose_mask21 + pred2 * (1 - transpose_mask21)) * transpose_cut2
    else:
        output1 = pred1 * transpose_cut1
        output2 = pred2 * transpose_cut2

    mixed_out_list.append(output1)
    mixed_out_list.append(output2)
    mixed_out = torch.stack(mixed_out_list)

    return mixed_out


def MixOutput4(pred, transpose_mask, transpose_mask_fix, transpose_cut, label):
    pred1 = pred[0:2, :, :, :]
    pred2 = pred[2:, :, :, :]

    label1 = label[0:2, :]
    label2 = label[2:, :]

    transpose_mask12 = transpose_mask[0:2, :, :, :]
    transpose_mask21 = transpose_mask[2:, :, :, :]

    transpose_mask_fix12 = transpose_mask_fix[0:2, :, :, :]
    transpose_mask_fix21 = transpose_mask_fix[2:, :, :, :]

    transpose_cut1 = transpose_cut[0:2, :, :, :]
    transpose_cut2 = transpose_cut[2:, :, :, :]

    mixed_out1 = MixOutput2(pred1, transpose_mask12, transpose_mask_fix12, transpose_cut1, label1)
    mixed_out2 = MixOutput2(pred2, transpose_mask21, transpose_mask_fix21, transpose_cut2, label2)
    mixed_out = torch.cat((mixed_out1, mixed_out2), 0)

    return mixed_out

def MixOutput8(pred, transpose_mask, transpose_mask_fix, transpose_cut, label):
    pred1 = pred[0:4, :, :, :]
    pred2 = pred[4:, :, :, :]

    label1 = label[0:4, :]
    label2 = label[4:, :]

    transpose_mask12 = transpose_mask[0:4, :, :, :]
    transpose_mask21 = transpose_mask[4:, :, :, :]

    transpose_mask_fix12 = transpose_mask_fix[0:4, :, :, :]
    transpose_mask_fix21 = transpose_mask_fix[4:, :, :, :]

    transpose_cut1 = transpose_cut[0:4, :, :, :]
    transpose_cut2 = transpose_cut[4:, :, :, :]

    mixed_out1 = MixOutput4(pred1, transpose_mask12, transpose_mask_fix12, transpose_cut1, label1)
    mixed_out2 = MixOutput4(pred2, transpose_mask21, transpose_mask_fix21, transpose_cut2, label2)
    mixed_out = torch.cat((mixed_out1, mixed_out2), 0)

    return mixed_out

def MixOutput16(pred, transpose_mask, transpose_mask_fix, transpose_cut, label):
    pred1 = pred[0:8, :, :, :]
    pred2 = pred[8:, :, :, :]

    label1 = label[0:8, :]
    label2 = label[8:, :]

    transpose_mask12 = transpose_mask[0:8, :, :, :]
    transpose_mask21 = transpose_mask[8:, :, :, :]

    transpose_mask_fix12 = transpose_mask_fix[0:8, :, :, :]
    transpose_mask_fix21 = transpose_mask_fix[8:, :, :, :]

    transpose_cut1 = transpose_cut[0:8, :, :, :]
    transpose_cut2 = transpose_cut[8:, :, :, :]

    mixed_out1 = MixOutput8(pred1, transpose_mask12, transpose_mask_fix12, transpose_cut1, label1)
    mixed_out2 = MixOutput8(pred2, transpose_mask21, transpose_mask_fix21, transpose_cut2, label2)
    mixed_out = torch.cat((mixed_out1, mixed_out2), 0)

    return mixed_out

def MixOutput32(pred, transpose_mask, transpose_mask_fix, transpose_cut, label):
    pred1 = pred[0:16, :, :, :]
    pred2 = pred[16:, :, :, :]

    label1 = label[0:16, :]
    label2 = label[16:, :]

    transpose_mask12 = transpose_mask[0:16, :, :, :]
    transpose_mask21 = transpose_mask[16:, :, :, :]

    transpose_mask_fix12 = transpose_mask_fix[0:16, :, :, :]
    transpose_mask_fix21 = transpose_mask_fix[16:, :, :, :]

    transpose_cut1 = transpose_cut[0:16, :, :, :]
    transpose_cut2 = transpose_cut[16:, :, :, :]

    mixed_out1 = MixOutput16(pred1, transpose_mask12, transpose_mask_fix12, transpose_cut1, label1)
    mixed_out2 = MixOutput16(pred2, transpose_mask21, transpose_mask_fix21, transpose_cut2, label2)
    mixed_out = torch.cat((mixed_out1, mixed_out2), 0)

    return mixed_out


def RoadPaste_batch_2(samples, targets_buffer, targets_mask):

    h = samples.shape[2]
    w = samples.shape[3]

    samples_list = []
    targets_list = []
    transport_mask = []
    transport_fix_mask = []
    label_list = []

    samples1 = samples[0, :, :, :]
    samples2 = samples[1, :, :, :]

    buffer1 = targets_buffer[0, :, :, :]
    buffer2 = targets_buffer[1, :, :, :]

    mask1 = targets_mask[0, :, :, :]
    mask2 = targets_mask[1, :, :, :]

    simi = compare_histograms(samples1, samples2)
    # print(simi)

    if simi > 0.5:
        label = np.ones((1))
        label = torch.tensor(label)

    # -------------------mix 1 to 2---------------------------------
        transport_mask12 = np.zeros((1, h, w), np.float32)
        transport_mask12 = torch.from_numpy(transport_mask12)

        transport_fix_mask12 = np.zeros((1, h, w), np.float32)
        transport_fix_mask12 = torch.from_numpy(transport_fix_mask12)
        transport_fix_mask12.cuda()

        z, x, y = torch.where(mask2 > 0)
        transport_mask12[0, x, y] = 1
        # print(samples1, transport_mask12, samples2)
        transport_mask12 = V(transport_mask12.cuda(), volatile=False)
        # print(samples1, samples2, transport_mask12)
        samples_mix12 = samples1 * (1 - transport_mask12) + samples2 * transport_mask12
        # z, x, y = torch.where(mask1 > 0)
        # samples_mix12[:, x, y] = samples1[:, x, y]/2 + samples_mix12[:, x, y]/2

        z, x, y = torch.where(buffer1 == 255)
        transport_fix_mask12[:, x, y] = 1
        transport_fix_mask12 = V(transport_fix_mask12.cuda(), volatile=False)
        # targets_mix12 = (mask2 * transport_mask12 + mask1 * (1 - transport_mask12)) * (1 - transport_fix_mask12) + transport_fix_mask12 * 255
        targets_mix12 = mask1 * (1 - transport_mask12) + mask2 * transport_mask12

        samples_list.append(samples_mix12)
        targets_list.append(targets_mix12)
        transport_mask.append(transport_mask12)
        transport_fix_mask.append(transport_fix_mask12)

    # -------------------mix 2 to 1---------------------------------
        transport_mask21 = np.zeros((1, h, w), np.float32)
        transport_mask21 = torch.from_numpy(transport_mask21)
        transport_mask21.cuda()
        transport_fix_mask21 = np.zeros((1, h, w), np.float32)
        transport_fix_mask21 = torch.from_numpy(transport_fix_mask21)
        transport_fix_mask21.cuda()

        z, x, y = torch.where(mask1 > 0)
        transport_mask21[0, x, y] = 1
        transport_mask21 = V(transport_mask21.cuda(), volatile=False)
        samples_mix21 = samples2 * (1 - transport_mask21) + samples1 * transport_mask21
        # z, x, y = torch.where(mask2 > 0)
        # samples_mix21[:, x, y] = samples2[:, x, y]/2 + samples_mix21[:, x, y]/2

        z, x, y = torch.where(buffer2 == 255)
        transport_fix_mask21[:, x, y] = 1
        transport_fix_mask21 = V(transport_fix_mask21.cuda(), volatile=False)
        # targets_mix21 = (mask1 * transport_mask21 + mask2 * (1 - transport_mask21)) * (1 - transport_fix_mask21) + transport_fix_mask21 * 255
        targets_mix21 = mask2 * (1 - transport_mask21) + mask1 * transport_mask21

        samples_list.append(samples_mix21)
        targets_list.append(targets_mix21)
        transport_mask.append(transport_mask21)
        transport_fix_mask.append(transport_fix_mask21)
        label_list.append(label)
        label_list.append(label)

        samples_out = torch.stack(samples_list)
        targets_out = torch.stack(targets_list)
        transport_mask_out = torch.stack(transport_mask)
        transport_fix_mask_out = torch.stack(transport_fix_mask)
        label_out = torch.stack(label_list)

        # print('out1:', label_out)

    else:
        label = np.zeros((1))
        label = torch.tensor(label)

        transport_mask12 = np.zeros((1, h, w), np.float32)
        transport_mask12 = torch.from_numpy(transport_mask12)
        transport_mask12.cuda()
        transport_mask21 = np.zeros((1, h, w), np.float32)
        transport_mask21 = torch.from_numpy(transport_mask21)
        transport_mask21.cuda()

        transport_fix_mask12 = np.zeros((1, h, w), np.float32)
        transport_fix_mask12 = torch.from_numpy(transport_fix_mask12)
        transport_fix_mask12.cuda()

        transport_fix_mask21 = np.zeros((1, h, w), np.float32)
        transport_fix_mask21 = torch.from_numpy(transport_fix_mask21)
        transport_fix_mask21.cuda()

        samples_list.append(samples1)
        targets_list.append(mask1)
        transport_mask.append(transport_mask12)
        transport_fix_mask.append(transport_fix_mask12)
        samples_list.append(samples2)
        targets_list.append(mask2)
        transport_mask.append(transport_mask21)
        transport_fix_mask.append(transport_fix_mask21)
        label_list.append(label)
        label_list.append(label)

        samples_out = torch.stack(samples_list).cuda()
        targets_out = torch.stack(targets_list).cuda()
        transport_mask_out = torch.stack(transport_mask).cuda()
        transport_fix_mask_out = torch.stack(transport_fix_mask).cuda()
        label_out = torch.stack(label_list)
        # print('out2:', label_out)

    return samples_out, targets_out, transport_mask_out, transport_fix_mask_out, label_out

def RoadPaste_batch_4(samples, targets_buffer, targets_mask):
    samples1 = samples[0:2, :, :, :]
    samples2 = samples[2:, :, :, :]

    buffer1 = targets_buffer[0:2, :, :, :]
    buffer2 = targets_buffer[2:, :, :, :]

    mask1 = targets_mask[0:2, :, :, :]
    mask2 = targets_mask[2:, :, :, :]

    samples_mix12, targets_mix12, transport_mask12, transport_fix_mask12, label1 = RoadPaste_batch_2(samples1, buffer1, mask1)
    samples_mix21, targets_mix21, transport_mask21, transport_fix_mask21, label2 = RoadPaste_batch_2(samples2, buffer2, mask2)

    samples_out = torch.cat((samples_mix12, samples_mix21), 0)
    targets_out = torch.cat((targets_mix12, targets_mix21), 0)
    transport_mask_out = torch.cat((transport_mask12, transport_mask21), 0)
    transport_fix_mask_out = torch.cat((transport_fix_mask12, transport_fix_mask21), 0)
    label_out = torch.cat((label1, label2), 0)

    return samples_out, targets_out, transport_mask_out, transport_fix_mask_out, label_out

def RoadPaste_batch_8(samples, targets_buffer, targets_mask):
    samples1 = samples[0:4, :, :, :]
    samples2 = samples[4:, :, :, :]

    buffer1 = targets_buffer[0:4, :, :, :]
    buffer2 = targets_buffer[4:, :, :, :]

    mask1 = targets_mask[0:4, :, :, :]
    mask2 = targets_mask[4:, :, :, :]

    samples_mix12, targets_mix12, transport_mask12, transport_fix_mask12, label1 = RoadPaste_batch_4(samples1, buffer1, mask1)
    samples_mix21, targets_mix21, transport_mask21, transport_fix_mask21, label2 = RoadPaste_batch_4(samples2, buffer2, mask2)

    samples_out = torch.cat((samples_mix12, samples_mix21), 0)
    targets_out = torch.cat((targets_mix12, targets_mix21), 0)
    transport_mask_out = torch.cat((transport_mask12, transport_mask21), 0)
    transport_fix_mask_out = torch.cat((transport_fix_mask12, transport_fix_mask21), 0)
    label_out = torch.cat((label1, label2), 0)

    return samples_out, targets_out, transport_mask_out, transport_fix_mask_out, label_out

def RoadPaste_batch_16(samples, targets_buffer, targets_mask):
    samples1 = samples[0:8, :, :, :]
    samples2 = samples[8:, :, :, :]

    buffer1 = targets_buffer[0:8, :, :, :]
    buffer2 = targets_buffer[8:, :, :, :]

    mask1 = targets_mask[0:8, :, :, :]
    mask2 = targets_mask[8:, :, :, :]

    samples_mix12, targets_mix12, transport_mask12, transport_fix_mask12, label1 = RoadPaste_batch_8(samples1, buffer1, mask1)
    samples_mix21, targets_mix21, transport_mask21, transport_fix_mask21, label2 = RoadPaste_batch_8(samples2, buffer2, mask2)

    samples_out = torch.cat((samples_mix12, samples_mix21), 0)
    targets_out = torch.cat((targets_mix12, targets_mix21), 0)
    transport_mask_out = torch.cat((transport_mask12, transport_mask21), 0)
    transport_fix_mask_out = torch.cat((transport_fix_mask12, transport_fix_mask21), 0)
    label_out = torch.cat((label1, label2), 0)

    return samples_out, targets_out, transport_mask_out, transport_fix_mask_out, label_out

def RoadPaste_batch_32(samples, targets_buffer, targets_mask):
    samples1 = samples[0:16, :, :, :]
    samples2 = samples[16:, :, :, :]

    buffer1 = targets_buffer[0:16, :, :, :]
    buffer2 = targets_buffer[16:, :, :, :]

    mask1 = targets_mask[0:16, :, :, :]
    mask2 = targets_mask[16:, :, :, :]

    samples_mix12, targets_mix12, transport_mask12, transport_fix_mask12, label1 = RoadPaste_batch_16(samples1, buffer1, mask1)
    samples_mix21, targets_mix21, transport_mask21, transport_fix_mask21, label2 = RoadPaste_batch_16(samples2, buffer2, mask2)

    samples_out = torch.cat((samples_mix12, samples_mix21), 0)
    targets_out = torch.cat((targets_mix12, targets_mix21), 0)
    transport_mask_out = torch.cat((transport_mask12, transport_mask21), 0)
    transport_fix_mask_out = torch.cat((transport_fix_mask12, transport_fix_mask21), 0)
    label_out = torch.cat((label1, label2), 0)

    return samples_out, targets_out, transport_mask_out, transport_fix_mask_out, label_out
