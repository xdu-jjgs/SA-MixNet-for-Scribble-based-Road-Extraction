import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('root', type=str)
    # parser.add_argument('split', type=str)

    # args = parser.parse_args()
    #
    # gts_path = os.path.join(args.root, args.split, 'mask')
    # preds_path = os.path.join(args.root, args.split, 'pred')
    preds_path = 'test_results'
    gts_path = './data/DeepGlobe/test/mask'

    matrix = np.zeros((2, 2), dtype=np.int64)
    for file in tqdm(os.listdir(gts_path), ascii=True):
        region_info = file[:-8]  # *mask.png
        gt_path = os.path.join(gts_path, file)
        # pred_path = os.path.join(preds_path, file.split('_')[0] + '_' + file.split('_')[1] + '_' + 'pred_b.png')
        pred_path = os.path.join(preds_path, file)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) // 255
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) // 255
        count = np.bincount(2 * gt.reshape(-1) + pred.reshape(-1), minlength=2 ** 2)
        matrix += count.reshape((2, 2))
    print('matrix: ', matrix)
    precisions = np.diag(matrix) / matrix.sum(axis=0)
    print('precisions: ', precisions)
    recalls = np.diag(matrix) / matrix.sum(axis=1)
    print('recalls: ', recalls)
    f1s = 2 * precisions * recalls / (precisions + recalls)
    print('f1s: ', f1s)
    intersection = np.diag(matrix)
    union = np.sum(matrix, axis=0) + np.sum(matrix, axis=1) - np.diag(matrix)
    ious = intersection / union
    print('ious: ', ious)


if __name__ == '__main__':
    main()
