import cv2
import numpy as np

from skimage import morphology

# label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) // 255


mask = cv2.imread(r'D:\Study\Work\Work-1-weaklyRoad\draw\10078660_15_1_mask.png', cv2.IMREAD_GRAYSCALE) // 255
osm = morphology.skeletonize(mask).astype(np.uint8) * 255

cv2.imwrite(r'D:\Study\Work\Work-1-weaklyRoad\draw\10078660_15_1_osm.png', osm)