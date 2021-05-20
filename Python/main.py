import os
import numpy as np
import cv2 as cv
import time

import PointOperators as po
import LinearFilter as lf
import RandomizedSelection as rs
import NonlinearFilter as nlf
import BinaryProcessing as bp


if __name__ == '__main__':
    print("current working space:", os.getcwd())
    print('running ......')

    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8

    img0 = cv.resize(cv.imread('../Data/imgs/apple1.jpg'), (128, 128))
    # img1 = cv.resize(cv.imread('../Data/imgs/1.jpg'), (512, 512))
    # img0 = cv.resize(cv.imread('../Data/imgs/girl1.jpg'), (256, 256))
    img0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

    # time0 = time.time()
    cv.imshow('img0', img0)
    # cv.imshow('image_transform0', pt.gainAndBias(img0, 2, 0).astype(np.uint8))
    # cv.imshow('image_transform1', pt.gainAndBias(img0, np.full(img0.shape, 1), 0).astype(np.uint8))
    # cv.imshow('image_transform2', pt.dyadic(img0, img1, 0.5).astype(np.uint8))
    # cv.imshow('ima0_gamma', po.gammaCorrect(img0, 2.2).astype(np.uint8))
    # cv.imshow('img0_hist_equalized', po.histogramEqualization(img0).astype(np.uint8))
    # cv.imshow('img0_block_equalized', po.blockHistgramEqualization(img0, (64, 64)).astype(np.uint8))
    # cv.imshow('img0_adp_block_equalized', po.locallyAdaptiveHistogramEqualization(img0, (256, 256)).astype(np.uint8))
    
    # time0 = time.time()
    # cv.imshow('img0_conv', lf.filter(img0, kernel, step=1, pad_size=1).astype(np.uint8))
    # print('conv:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_conv_2d', lf.filter_2d(img0, kernel, step=1, pad_size=1).astype(np.uint8))
    # print('conv_2d:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_cv_conv_2d', cv.filter2D(img0, -1, kernel))
    # print('conv_2d:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_gaussian', lf.gaussianFilter(img0, ksize=3, sigma=0.8).astype(np.uint8))
    # print('gaussian:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_fastGaussian', lf.fastGaussianFilter(img0, ksize=3, sigma=0.8).astype(np.uint8))
    # print('fastGaussian:', time.time()-time0)
    # cv.imshow('img0_laplacian', lf.laplacianFilter(img0).astype(np.uint8))
    # cv.imshow('img0_cv_lalacian', cv.Laplacian(img0, -1))
    # cv.imshow('img0_laplacianOfGaussian', lf.laplacianOfGaussianFilter(img0).astype(np.uint8))
    # time0 = time.time()
    # cv.imshow('img0_separable_filter', lf.separableFilter(img0, kernel, step=1, pad_size=1).astype(np.uint8))
    # print('separable conv:', time.time()-time0)

    # a = np.ones((5, 5, 3))
    # s = lf.summedAreaTable(a)
    # print(lf.areaComputation(s, 1, 1, 3, 3))

    # a = np.arange(17)
    # print(rs.randomizedSelect(a, 0, a.shape[0]-1, 9))

    # time0 = time.time()
    # cv.imshow('img0_mean_filter', lf.meanFilter(img0, 3).astype(np.uint8))
    # print('mean filter:', time.time()-time0)

    # time0 = time.time()
    # cv.imshow('img0_median_filter', nlf.medianFilter(img0, 3).astype(np.uint8))
    # print('median filter:', time.time()-time0)

    # time0 = time.time()
    # cv.imshow('img0_weighted_median', nlf.weightedMedianFilter(img0, np.random.randint(1, 10, size=(img0.shape[0], img0.shape[1])), 3).astype(np.uint8))
    # print('weighted median:', time.time()-time0)

    # time0 = time.time()
    img0 = bp.binarize(img0, np.mean(img0), 255).astype(np.uint8)
    cv.imshow('img0_binarize', img0)
    # print('binarize:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_dilate', bp.dilate(img0, 3))
    # print('dilate:', time.time()-time0)
    # cv.imshow('img0_cv_dilate', cv.dilate(img0, np.ones((3, 3))))
    # time0 = time.time()
    # cv.imshow('img0_erode', bp.erode(img0, 3))
    # print('erode:', time.time()-time0)
    # cv.imshow('img0_cv_erode', cv.erode(img0, np.ones((3, 3))))
    # time0 = time.time()
    # cv.imshow('img0_thresh_morphing', bp.threshMorphing(img0, 3, 1))
    # print('thresh_morphing:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_majority', bp.majority(img0, 3))
    # print('majority:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_opening', bp.opening(img0, 3))
    # print('opening:', time.time()-time0)
    # time0 = time.time()
    # cv.imshow('img0_closing', bp.closing(img0, 3))
    # print('closing:', time.time()-time0)
    # time0 = time.time()
    # print('img0_city_block_distance:', bp.cityBlockDistance(img0))
    # print('city block distance:', time.time()-time0)
    # time0 = time.time()
    # comp_cnt, conn_comp = bp.connectedComponents(img0, 200)
    # print('connected components:', time.time()-time0)
    # print('img0_connected_components:', comp_cnt, conn_comp)
    # conn_comp_img = (conn_comp/np.max(conn_comp) * 255).astype(np.uint8)
    # cv.imshow('connected components', conn_comp_img)
    

    cv.waitKey(0)

    print('...... done.')
