import numpy as np
from numpy.lib.polynomial import RankWarning

def buildGaussianPyramid(img:np.ndarray, octvs, intvls, sigma0):
    gaussian_pyr = []
    for o in range(octvs):
        gaussian_pyr.append([gaussianBlur(img, sigma=sigma0 * pow(2, o+s/intvls)) for s in range(intvls)])
        img = downsample(img)
    return gaussian_pyr

def gaussianBlur(img:np.ndarray, sigma=0.8):
    ksize = 2*int(3*sigma) + 1
    kernel = generateGaussianKernel(ksize, sigma)
    return conv2d(img, kernel, step=1, pad_size=int((ksize-1)/2))

def generateGaussianKernel(ksize=3, sigma=0.8):
    kernel = np.zeros((ksize, ksize))
    origin = int(ksize / 2)
    for y in range(ksize):
        for x in range(ksize):
            kernel[y, x] = np.e**(-((x-origin)**2 + (y-origin)**2)/(2*sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def conv2d(img:np.ndarray, kernel:np.ndarray, step=1, pad_size=0):
    img_pad = np.pad(img, [(pad_size,)]*kernel.ndim + [(0,)]*(img.ndim-kernel.ndim))
    shape_out = tuple([int((2*pad_size+w-kw)/step + 1) for w, kw in zip(img.shape, kernel.shape)])
    shape_rem = tuple([img.shape[k] for k in range(kernel.ndim, img.ndim)])
    shape_out += shape_rem
    img_out = np.zeros(shape_out)
    ker = np.expand_dims(kernel, axis=tuple(range(kernel.ndim, img_pad.ndim))) if kernel.ndim < img_pad.ndim else kernel
    for i, y0 in enumerate(range(0, img_pad.shape[0]-kernel.shape[0]+1, step)):
        for j, x0 in enumerate(range(0, img_pad.shape[1]-kernel.shape[1]+1, step)):
            img_out[i, j] = np.sum((img_pad[y0:y0+kernel.shape[0], x0:x0+kernel.shape[1]] * ker).reshape((-1,) + shape_rem), axis=0)
            # img_out[i, j] = np.sum((img_pad[tuple([range(x, x+w) for x, w in zip((y0, x0), kernel.shape)])] * ker).reshape((-1,) + shape_rem), axis=0)
    return img_out

elewise_mul_as_int = lambda a,b : tuple([int(np.ceil(x*y)) for x,y in zip(a,b)])

def downsample(img:np.ndarray, rt=(0.5, 0.5)):
    shape_pre = elewise_mul_as_int(img.shape[0:len(rt)], rt)
    shape_rem = tuple(img.shape[len(rt):img.ndim])
    img_out = np.zeros((shape_pre+shape_rem))
    kh, kw = img.shape[0]/img_out.shape[0], img.shape[1]/img_out.shape[1]
    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):
            img_out[i, j] = np.average(img[int(np.floor(i*kh)):int(np.ceil((i+1)*kh)), int(np.floor(j*kw)):int(np.ceil((j+1)*kw))], axis=(0, 1))
    return img_out

def buildDiffrenceOfGaussain(gaussian_pyr):
    dog_pyr = [[octv[i+1]-octv[i] for i in range(len(octv)-1)] for octv in gaussian_pyr]
    return dog_pyr

def scaleSpaceExtrema(dog_pyr, octvs, intvls, contr_thr, curv_thr):
    extremum_arr = []
    prelim_contr_thr = 0.5 * contr_thr / intvls
    for o in range(len(dog_pyr)):
        h, w = dog_pyr[o][0].shape[0:2]
        for i in range(len(dog_pyr[o])):
            for r in range(1, h-1):
                for c in range(1, w-1):
                    if np.abs(dog_pyr[o][i][r, c])>prelim_contr_thr and isExtremum(dog_pyr, o, i, r, c):
                        extremum_arr.append(dog_pyr[o][i][r, c])
    return extremum_arr

def isExtremum(dog_pyr, o, i, r, c):
    return ((3*r+c == np.argmin(dog_pyr[o][i][r-1:r+2, c-1:c+2])
            and dog_pyr[o][i][r,c] <= np.min(dog_pyr[o][i-1][r-1:r+2, c-1:c+2])
            and dog_pyr[o][i][r,c] <= np.min(dog_pyr[o][i+1][r-1:r+2, c-1:c+2]))
            or (3*r+c == np.argmax(dog_pyr[o][i][r-1:r+2, c-1:c+2])
            and dog_pyr[o][i][r,c] >= np.max(dog_pyr[o][i-1][r-1:r+2, c-1:c+2])
            and dog_pyr[o][i][r,c] >= np.max(dog_pyr[o][i+1][r-1:r+2, c-1:c+2])))
