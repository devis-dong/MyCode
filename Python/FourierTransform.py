import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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

#get the dft_matrix
def dft_matrix(N):
	i,j = np.meshgrid(np.arange(N), np.arange(N))
	omega = np.exp(-2j*np.pi/N)
	w = np.power(omega,i*j)
	return w

def dft2d(image,flags=0):
	h,w = image.shape[:2]
	# image_shift = np.zeros((h,w),np.uint8)
	output = np.zeros((h,w), np.complex)
	# for x in xrange(h):
	# 	for y in xrange(w):
	# 		image_shift[x,y] = image[x,y]*(-1)**(x+y)
	output = dft_matrix(h).dot(image).dot(dft_matrix(w))
	return output

def fftshift(dft):
    h, w = dft.shape
    s_dft = np.zeros_like(dft)
    s_dft[0:h-int(h/2):, :] = dft[int(h/2):h, :]
    s_dft[h-int(h/2):h, :] = dft[0:int(h/2), :]
    tmp = s_dft[:, 0:int(w/2)].copy()
    s_dft[:, 0:w-int(w/2)] = s_dft[:, int(w/2):w]
    s_dft[:, w-int(w/2):w] = tmp
    return s_dft
