import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pixel_transform as pt


if __name__ == '__main__':
    print("current working space:", os.getcwd())

    img = mpimg.imread('../Data/imgs/0.jpeg')
    plt.figure('image_origin')
    plt.imshow(img)

    img = pt.multiply_and_add_with_a_constant(img, 2, 9)
    plt.figure('image_transform')
    plt.imshow(img)
    plt.show()
