import os
import numpy as np
import cv2 as cv

import pixel_transform as pt


if __name__ == '__main__':
    print("current working space:", os.getcwd())

    img0 = cv.resize(cv.imread('../Data/imgs/0.jpg'), (1024, 1024))
    img1 = cv.resize(cv.imread('../Data/imgs/1.jpg'), (1024, 1024))

    cv.imshow('img0', img0)
    # cv.imshow('image_transform0', pt.gainAndBias(img0, 2, 0).astype(np.uint8))
    # cv.imshow('image_transform1', pt.gainAndBias(img0, np.full(img0.shape, 1), 0).astype(np.uint8))
    # cv.imshow('image_transform2', pt.dyadic(img0, img1, 0.5).astype(np.uint8))
    cv.imshow('ima0_gamma', pt.gammaCorrect(img0, 2.2).astype(np.uint8))

    cv.waitKey(0)