import numpy as np
import scipy as sp
import cv2
from scipy.fftpack import fft, dct
from numpy import r_

#Função DCT
def dct2(a):
    return sp.fftpack.dct(sp.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho');

#Função DCT Inversa
def idct2(a):
    return sp.fftpack.idct(sp.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho');


#Ler Imagem
cachorro = cv2.imread('CachorroCasamento3.jpg')

#Transformar a Imagem em Escalas de Cinza
grayCachorro = cv2.cvtColor(cachorro, cv2.COLOR_BGR2GRAY);

#Fatiamento da Imagem (Block8x8)
imagegray = np.zeros(grayCachorro.shape, np.int8);
imagegray[:,:] = grayCachorro[:,:] - 128;
imageDCT = np.zeros(grayCachorro.shape);
for i in range (0, grayCachorro.shape[0], 8):
    for j in range (0, grayCachorro.shape[1], 8):
        imageDCT[i:(i+8), j:(j+8)] = dct2(imagegray[i:(i+8), j:(j+8)]);

#Função de Quantização
def quantization(Q):

    QA = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]]);

    K = 48;
    Q = K * QA;
    return Q;



cv2.imshow('Original', grayCachorro.astype(np.uint8));
cv2.imshow('Block8X8', imageDCT.astype(np.uint8));
cv2.waitKey(0);
cv2.destroyAllWindows();

