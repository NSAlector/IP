import numpy as np
import cv2 as cv

class DFT:
    def __init__(self):
        pass

    def dft(self, img):
        dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        return dft_shift

    def idft(self, dft_shift):
        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv.idft(f_ishift) / f_ishift.shape[0] / f_ishift.shape[1]
        img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        #img_back = img_back[:,:,0]
        return img_back