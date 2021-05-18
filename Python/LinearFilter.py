import numpy as np
import itertools

from numpy.core.fromnumeric import shape

vec_add = lambda a,b : tuple([x+y for x,y in zip(a,b)])

def convolve(img:np.ndarray, kernel:np.ndarray, step=1, pad_size=0):
    pad_width = [(0, 0)] * img.ndim
    for k in range(0, kernel.ndim):
        pad_width[k] = (pad_size, pad_size)
    img_pad = np.pad(img, pad_width)
    k_idx = iter_idx(kernel.shape)
    o_idx = iter_idx(tuple([img_pad.shape[i]-kernel.shape[i]+1 for i in range(kernel.ndim)]), step)
    shape_out = [int((2*pad_size+w-kw)/step + 1) for w, kw in zip(img.shape, kernel.shape)]
    i_idx = iter_idx(shape_out)
    for k in range(kernel.ndim, img.ndim):
        shape_out.append(img.shape[k])
    shape_out = tuple(shape_out)
    img_out = np.zeros(shape_out)
    for i, o in zip(i_idx, o_idx):
        for k in k_idx:
            img_out[i] += img_pad[vec_add(o, k)]*kernel[k]
            # print(img_pad[vec_add(o, i)]*kernel[i])
    return img_out

def iter_idx(shape, step=1):
    return list(itertools.product(*[range(0, i, step) for i in shape]))



    
# a = np.arange(12).reshape(2, 2, 3)
# b = np.arange(48).reshape(2, 3, 4, 2)
# k_idx = iter_idx(a.shape)
# o_idx = iter_idx([b.shape[i] for i in range(a.ndim)], 1)
# for o in o_idx:
#     for i in k_idx:
#         print(b[vec_add(o, i)])
