import os
import numpy as np
import cv2 as cv
import time

import PointOperators as po
import LinearFilter as lf


if __name__ == '__main__':
    print("current working space:", os.getcwd())

    img0 = cv.resize(cv.imread('../Data/imgs/2.jpg'), (512, 512))
    # img1 = cv.resize(cv.imread('../Data/imgs/1.jpg'), (512, 512))
    # img0 = cv.resize(cv.imread('../Data/imgs/girl1.jpg'), (256, 256))

    # cv.imshow('img0', img0)
    # cv.imshow('image_transform0', pt.gainAndBias(img0, 2, 0).astype(np.uint8))
    # cv.imshow('image_transform1', pt.gainAndBias(img0, np.full(img0.shape, 1), 0).astype(np.uint8))
    # cv.imshow('image_transform2', pt.dyadic(img0, img1, 0.5).astype(np.uint8))
    # cv.imshow('ima0_gamma', po.gammaCorrect(img0, 2.2).astype(np.uint8))

    img0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    cv.imshow('img0_gray', img0)
    # img0_hist_equalized = po.histogramEqualization(img0_gray)
    # cv.imshow('img0_hist_equalized', img0_hist_equalized)
    # img0_block_equalized = po.blockHistgramEqualization(img0_gray, (64, 64))
    # cv.imshow('img0_block_equalized', img0_block_equalized)
    # img0_adp_block_equalized = po.locallyAdaptiveHistogramEqualization(img0_gray, (256, 256))
    # cv.imshow('img0_adp_block_equalized', img0_adp_block_equalized)

    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    img0_conv = lf.convolve(img0, kernel, step=1, pad_size=1).astype(np.uint8)
    cv.imshow('img0_conv', img0_conv)

    cv.waitKey(0)
