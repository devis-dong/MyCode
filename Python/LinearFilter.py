import numpy as np
import itertools

from numpy.core.fromnumeric import ndim, shape
from numpy.lib import RankWarning, pad

vec_add = lambda a,b : tuple([x+y for x,y in zip(a,b)])

def convolve(img:np.ndarray, kernel:np.ndarray, step=1, pad_size=0):
    if 2 == kernel.ndim:
        return convolve_2d(img,kernel, step, pad_size)
    img_pad = np.pad(img, [(pad_size,)]*kernel.ndim + [(0,)]*(img.ndim-kernel.ndim))
    k_idx = iterIdx(kernel.shape)
    o_idx = iterIdx(tuple([img_pad.shape[i]-kernel.shape[i]+1 for i in range(kernel.ndim)]), step)
    shape_out = [int((2*pad_size+w-kw)/step + 1) for w, kw in zip(img.shape, kernel.shape)]
    i_idx = iterIdx(shape_out)
    for k in range(kernel.ndim, img.ndim):
        shape_out.append(img.shape[k])
    img_out = np.zeros(tuple(shape_out))
    for i, o in zip(i_idx, o_idx):
        for k in k_idx:
            img_out[i] += img_pad[vec_add(o, k)]*kernel[k]
            # print(img_pad[vec_add(o, i)]*kernel[i])
    return img_out

def convolve_2d(img:np.ndarray, kernel:np.ndarray, step=1, pad_size=0):
    img_pad = np.pad(img, [(pad_size,)]*kernel.ndim + [(0,)]*(img.ndim-kernel.ndim))
    shape_out = [int((2*pad_size+w-kw)/step + 1) for w, kw in zip(img.shape, kernel.shape)]
    for k in range(kernel.ndim, img.ndim):
        shape_out.append(img.shape[k])
    img_out = np.zeros(tuple(shape_out))
    for i, y0 in enumerate(range(0, img_pad.shape[0]-kernel.shape[0]+1, step)):
        for j, x0 in enumerate(range(0, img_pad.shape[1]-kernel.shape[1]+1, step)):
            for y in range(kernel.shape[0]):
                for x in range(kernel.shape[1]):
                    img_out[i, j] += img_pad[y0+y, x0+x] * kernel[y, x]
    return img_out

# def padImg(img:np.ndarray, pad_size=None):
#     if pad_size is None:
#         return img
#     pad_width = [(w, w) for w in pad_size] + [(0, 0)]*(img.ndim-len(pad_size))
#     img_pad = np.pad(img, pad_width)
#     return img_pad

def iterIdx(shape, step=1):
    return list(itertools.product(*[range(0, i, step) for i in shape]))

def generateGaussianKernel(ksize=3, sigma=0.8):
    kernel = np.zeros((ksize, ksize))
    origin = int(ksize / 2)
    for y in range(ksize):
        for x in range(ksize):
            kernel[y, x] = np.e**(-((x-origin)**2 + (y-origin)**2)/(2*sigma**2))
    kernel = np.floor(kernel / kernel[0, 0])
    kernel /= np.sum(kernel)
    return kernel

def generateLaplacianKernel():
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / 8
    return kernel

def gaussianFilter(img:np.ndarray, ksize=3, sigma=0.8):
    kernel = generateGaussianKernel(ksize, sigma)
    return convolve(img, kernel, step=1, pad_size=1)

def generateGaussianVector(ksize=3, sigma=0.8):
    vec = np.zeros((ksize))
    origin = int(ksize / 2)
    for x in range(ksize):
        vec[x] = (np.e**(-(x-origin)**2))/(2*sigma**2)
    vec = np.floor(vec / vec[0])
    vec /= np.sum(vec)
    return vec

def fastGaussianFilter(img:np.ndarray, ksize=3, sigma=0.8):
    vec = generateGaussianVector(ksize, sigma)
    img_conv = np.pad(img, [(1,)]*2+[(0,)]*(img.ndim-2))
    img_conv = convolve(img, vec.reshape(1, -1), step=1, pad_size=0)
    img_conv = convolve(img, vec.reshape(-1, 1), step=1, pad_size=0)
    return img_conv

def laplacianFilter(img:np.ndarray):
    kernel = generateLaplacianKernel()
    return convolve(img, kernel, step=1, pad_size=1)

def generateLaplacianOfGaussianKernel(ksize=3, sigma=0.8):
    kernel = np.zeros((ksize, ksize))
    origin = int(ksize / 2)
    for y in range(ksize):
        for x in range(ksize):
            kernel[y, x] = ((x-origin)**2 + (y-origin)**2 - 2*sigma**2)/(sigma**4) * np.e**(-((x-origin)**2 + (y-origin)**2)/(2*sigma**2))
    kernel = np.floor(kernel / kernel[0, 0])
    kernel /= np.sum(kernel)
    return kernel

def laplacianOfGaussianFilter(img:np.ndarray, ksize=3, sigma=0.8):
    kernel = generateLaplacianOfGaussianKernel(ksize, sigma)
    return convolve(img, kernel, step=1, pad_size=1)

def fastLaplacianOfGaussianFilter(img:np.ndarray, ksize=3, sigma=0.8):
    kernel = generateLaplacianOfGaussianKernel(ksize, sigma)
    return convolve(img, kernel, step=1, pad_size=1)

def separableFilter(img:np.ndarray, kernel:np.ndarray, step=1, pad_size=0):
    U, s, VT = np.linalg.svd(kernel)
    u, v = (s[0]**0.5)*U[:, [0]], (s[0]**0.5)*VT[[0], :]
    img_conv = np.pad(img, [(pad_size,)]*kernel.ndim+[(0,)]*(img.ndim-kernel.ndim))
    img_conv = convolve(img_conv, v, step=step, pad_size=0)
    img_conv = convolve(img_conv, u, step=step, pad_size=0)
    return img_conv



    
# a = np.arange(12).reshape(2, 2, 3)
# b = np.arange(48).reshape(2, 3, 4, 2)
# k_idx = iter_idx(a.shape)
# o_idx = iter_idx([b.shape[i] for i in range(a.ndim)], 1)
# for o in o_idx:
#     for i in k_idx:
#         print(b[vec_add(o, i)])
