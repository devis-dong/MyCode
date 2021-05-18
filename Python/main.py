import os
import numpy as np
import cv2 as cv
import time

import PointOperators as po
import LinearFilter as lf


if __name__ == '__main__':
    print("current working space:", os.getcwd())
    print('running ......')
    img0 = cv.resize(cv.imread('../Data/imgs/2.jpg'), (512, 512))
    # img1 = cv.resize(cv.imread('../Data/imgs/1.jpg'), (512, 512))
    # img0 = cv.resize(cv.imread('../Data/imgs/girl1.jpg'), (256, 256))
    # img0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

    # time0 = time.time()
    # cv.imshow('img0', img0)
    # cv.imshow('image_transform0', pt.gainAndBias(img0, 2, 0).astype(np.uint8))
    # cv.imshow('image_transform1', pt.gainAndBias(img0, np.full(img0.shape, 1), 0).astype(np.uint8))
    # cv.imshow('image_transform2', pt.dyadic(img0, img1, 0.5).astype(np.uint8))
    # cv.imshow('ima0_gamma', po.gammaCorrect(img0, 2.2).astype(np.uint8))
    # cv.imshow('img0_hist_equalized', po.histogramEqualization(img0).astype(np.uint8))
    # cv.imshow('img0_block_equalized', po.blockHistgramEqualization(img0, (64, 64)).astype(np.uint8))
    # cv.imshow('img0_adp_block_equalized', po.locallyAdaptiveHistogramEqualization(img0, (256, 256)).astype(np.uint8))
    # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    # time0 = time.time()
    # cv.imshow('img0_conv', lf.convolve(img0, kernel, step=1, pad_size=1).astype(np.uint8))
    # print('conv:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_conv_2d', lf.convolve_2d(img0, kernel, step=1, pad_size=1).astype(np.uint8))
    # print('conv_2d:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_gaussian', lf.gaussianFilter(img0, ksize=3, sigma=0.8).astype(np.uint8))
    # print('gaussian:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_fastGaussian', lf.fastGaussianFilter(img0, ksize=3, sigma=0.8).astype(np.uint8))
    # print('fastGaussian:', time.time()-time0)
    cv.imshow('img0_laplacian', lf.laplacianFilter(img0).astype(np.uint8))
    # cv.imshow('img0_cv_lalacian', cv.Laplacian(img0, -1))
    cv.imshow('img0_laplacianOfGaussian', lf.laplacianOfGaussianFilter(img0).astype(np.uint8))
    # time0 = time.time()
    # cv.imshow('img0_separable_filter', lf.separableFilter(img0, kernel, step=1, pad_size=1).astype(np.uint8))
    # print('separable conv:', time.time()-time0)

    cv.waitKey(0)

    print('...... done.')
