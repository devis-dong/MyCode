import os
import numpy as np
import cv2 as cv
import time

import SIFT

print("current working space:", os.getcwd())
print('running ......')

# img = cv.resize(cv.imread('../Data/imgs/scene0.jpg'), (256, 256))
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('origin img', img)
# # cv.imshow('gaussina blur', SIFT.gaussianBlur(img).astype(np.uint8))
# time0 = time.time()
# gaussian_pyr = SIFT.buildGaussianPyramid(img, octvs=int(np.log2(min(img.shape[0:2]))-3), intvls=4, sigma=1.6)
# print("buildGaussianPyramid:", time.time()-time0)
# for i in range(len(gaussian_pyr)):
#     print(i, gaussian_pyr[i][0].shape)
#     cv.imshow('octv%d' % i, gaussian_pyr[i][0].astype(np.uint8))
# dog = SIFT.buildDiffrenceOfGaussain(gaussian_pyr)
# for i in range(len(dog)):
#     cv.imshow('dog%s'%i, dog[i][0].astype(np.uint8))


# cv.waitKey(0)

print('done...!!')