import numpy as np
from numpy import ma
import os
import matplotlib

import cv2
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from data_utils import affinity_utils

from skimage import data
from data_utils import sknw

def get_OSM_points_edges(image_osm):
    a = image_osm

    skeleton = np.zeros((512, 512, 3))
    skeleton[:, :, 1] = a
    graph = sknw.build_sknw(skeleton)
    nodes = graph.nodes
    ps = np.array([nodes[i]['o'] for i in nodes])
    edge = graph.edges

    x, y = np.where(a == 255)
    dis = np.zeros((np.shape(ps)[0], 2))

    for i in range(np.shape(ps)[0]):
        X = 500
        Y = 500
        distance = 2000
        if (ps[i, 0] % 1) > 0:
            for j in range(np.shape(x)[0]):
                d = abs(x[j] - ps[i, 0]) + abs(y[j]  - ps[i, 1])
                if d < distance:
                    distance = d
                    X, Y = x[j], y[j]
        else:
            if a[int(ps[i, 0]), int(ps[i, 1])] > 0:
                X, Y = int(ps[i, 0]), int(ps[i, 1])
            else:
                for j in range(np.shape(x)[0]):
                    d = abs(x[j] - ps[i, 0]) + abs(y[j]  - ps[i, 1])
                    if d < distance:
                        distance = d
                        X, Y = x[j], y[j]
        dis[i, 0] = int(X)
        dis[i, 1] = int(Y)
    return dis, edge, graph

def get_keypoint(image_root, n):

    image_osm = cv2.imread(image_root, cv2.IMREAD_GRAYSCALE)

    dis, edge, graph = get_OSM_points_edges(image_osm)

    # a = np.array(edge)
    # print(a)
    # print(dis)

    osm_point = np.zeros((512, 512))

    point_matrix = np.zeros((1,3))
    # print(point_matrix)
    n = int(512 / np.power(n, 4/7)) + 1

    for (s, e) in graph.edges():
        ps = graph[s][e]['pts']
        # print(ps)
        # print(np.shape(ps))
        for i in range(np.shape(ps)[0]):
            if (i % n == 0):
                osm_point[ps[i, 0], ps[i, 1]] = 1
                a = np.array([[0, ps[i, 0], ps[i, 1]]])
                point_matrix = np.vstack([point_matrix, a])
        osm_point[ps[np.shape(ps)[0]-1, 0], ps[np.shape(ps)[0]-1, 1]] = 1
        point_matrix[0, 1], point_matrix[0, 2] = ps[np.shape(ps)[0]-1, 0], ps[np.shape(ps)[0]-1, 1]

    point_matrix = point_matrix.astype(int)
    point_matrix_copy = point_matrix
    # print(n)

    for i in range(np.shape(dis)[0]):
        for j in range(np.shape(point_matrix)[0]):
            if (abs(dis[i, 0] - point_matrix[j, 1]) + abs(dis[i, 1] - point_matrix[j, 2]) >= n / 2):
                continue
            else:
                # point_matrix_copy = np.delete(point_matrix_copy, j-k, axis=0)
                point_matrix_copy[j, 1] = 777

    x = np.where(point_matrix_copy[:, 1] == 777)
    point_matrix_copy = np.delete(point_matrix_copy, x, axis=0)

    return point_matrix_copy


# point = get_keypoint(r'D:\hh\ScRoadExtractor-master\data\deepglobe_512\train\osm\104_2_osm.png', 1024)
# image = np.zeros((512, 512, 3))
#
# image[point[:, 1], point[:, 2], 0] = 255
# image[point[:, 1], point[:, 2], 1] = 255
# image[point[:, 1], point[:, 2], 2] = 255
#
# print(np.shape(point))
# # print(point)
# # cv2.imwrite(r'G:\113_4_point.png', image)
