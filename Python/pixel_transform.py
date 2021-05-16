import numpy as np

def gainAndBias(img, a, b):
    return a * img + b

def dyadic(img0, img1, alpha):
    return (1 - alpha) * img0 + alpha * img1

def gammaCorrect(img, gamma):
    return img ** (1.0/gamma)



    

