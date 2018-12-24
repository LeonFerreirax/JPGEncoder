import inline as inline
import numpy as np
import scipy as sp
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from scipy import misc
from scipy.fftpack import fft, dct

# #Ler a imagem
# image = cv2.imread('CachorroCasamento2.jpg');
#
# #Converter a Imagem em Escalas de Cinza
# grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
#
# #Salvar imagem em escala de cinza
# cv2.imwrite('graycacchorrocasamento.jpg', grayImage);
#
#Ler Imagem em cinza
graycachorro = cv2.imread('graycacchorrocasamento.jpg');
#
# #Mostrar a Imagem
# cv2.imshow('image', image);
# cv2.imshow('testecinza', graycachorro);
#
# cv2.waitKey(0);
# cv2.destroyAllWindows();

#Definição da DCT e da DCT Inversa
def dct2(a):
    return sp.fftpack.dct(sp.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho');
def idct2(a):
    return sp.fftpack.idct(sp.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho');

#Teste da função
DCTImagem = dct2(graycachorro);
IDCTImagem = idct2(DCTImagem);
cv2.imshow('testecinza', graycachorro);
cv2.imshow('testedct', DCTImagem.astype(np.uint8));
cv2.imshow('testeidct', IDCTImagem.astype(np.uint8));
cv2.waitKey(0);
cv2.destroyAllWindows();
