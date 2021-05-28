import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy.lib.index_tricks import nd_grid

def showSpectrum(f_img:np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ys, xs = np.meshgrid(range(f_img.shape[0]), range(f_img.shape[1]))
    surf = ax.plot_surface(xs, ys, f_img, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    # ax.set_xlim(0, f_img.shape[1])
    # ax.set_ylim(0, f_img.shape[0])
    # ax.set_zlim(np.min(f_img), np.max(f_img))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title("Surface plot", weight='bold', size=20)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=1, aspect=7)
    plt.show()

#get the dft matrix
def dft_matrix(N):
	return np.power(np.exp(-2j*np.pi/N), np.outer(np.arange(N), np.arange(N)))

# def fftshift(dft):
#     h, w = dft.shape
#     s_dft = np.zeros_like(dft)
#     s_dft[0:h-int(h/2):, :] = dft[int(h/2):h, :]
#     s_dft[h-int(h/2):h, :] = dft[0:int(h/2), :]
#     tmp = s_dft[:, 0:int(w/2)].copy()
#     s_dft[:, 0:w-int(w/2)] = s_dft[:, int(w/2):w]
#     s_dft[:, w-int(w/2):w] = tmp
#     return s_dft

def dft2d(img:np.ndarray, shift_flag=False):
    h, w = img.shape[0:2]
    if shift_flag:
        j, i = np.meshgrid(np.arange(w), np.arange(h))
        return dft_matrix(h).dot(img * np.power(-1, i+j)).dot(dft_matrix(w)) / (h*w)
    else:
        return dft_matrix(h).dot(img).dot(dft_matrix(w)) / (h*w)

def idft_matrix(N):
	w = np.power(np.exp(2j*np.pi/N), np.outer(np.arange(N), np.arange(N)))
	return w

def idft2d(img:np.ndarray, shift_flag=False):
    h, w = img.shape[0:2]
    if shift_flag:
        j, i = np.meshgrid(np.arange(w), np.arange(h))
        return idft_matrix(h).dot(img).dot(idft_matrix(w)) * np.power(-1, i+j)
    else:
        return idft_matrix(h).dot(img).dot(idft_matrix(w))

def band_pass(img:np.ndarray, kh0, kh1, kw0, kw1):
    img_out = np.zeros_like(img)
    img_out[kh0:kh1, kw0:kw1] = img[kh0:kh1, kw0:kw1]
    return img_out

def dct_matrix(N):
    return np.cos(np.pi * np.outer(np.linspace(0, N-1, N), np.linspace(0.5, N-0.5, N)) / N)

def dct2d(img:np.ndarray):
    h, w = img.shape[0:2]
    return dct_matrix(h).dot(img).dot(dct_matrix(w).T) / (h*w)

def idct2d(img:np.ndarray):
    h, w = img.shape[0:2]
    return (dct_matrix(h).T).dot(img).dot(dct_matrix(w))

def reverseBit(a):
    b = 0
    while a != 0:
        b = b << 1
        b = b | (a & 1)
        a = a >> 1
    return b

def fft(x:np.ndarray):
    F = np.array([x[reverseBit(i)] for i in range(x.shape[0])], dtype=np.complex)
    k = 1
    while k != x.shape[0]:
        F_tmp = np.zeros_like(F, np.complex)
        half_k = k
        k *= 2
        Wk = np.exp(-2j*np.pi/k)
        for i in range(0, F.shape[0], k):
            for j in range(k):
                F_tmp[i+j] = F[i+j%half_k] + (Wk**j) * F[i+half_k+j%half_k]
        F = F_tmp.copy()
    return F/x.shape[0]

def fft2d(img:np.ndarray):
    return fft(fft(img.T).T)


