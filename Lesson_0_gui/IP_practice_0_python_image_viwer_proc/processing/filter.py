import numpy as np
import cv2 as cv
from copy import deepcopy

import dft_image.dft


class Filter(object):
     def __init__(self):
         pass
     def processing(self, img):
         pass

# get magnitude spectrum
class FilterVis(Filter):
    def __init__(self):
        pass

    def processing(self, dft_shift):
        magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        return magnitude_spectrum

class LowPassFilter(Filter):
    def __init__(self, r):
        self.r = r

    def processing(self, dft_shift):
        mask = np.zeros((dft_shift.shape[0], dft_shift.shape[1], 2), np.float32)
        centerX, centerY = (int(dft_shift.shape[0] / 2), int(dft_shift.shape[1] / 2))
        mask[centerX -self.r:centerX + self.r, centerY - self.r:centerY + self.r, :] = 1
        dft_shift = dft_shift*mask

        return dft_shift

class HighPassFilter(Filter):
    def __init__(self, r):
        self.r = r

    def processing(self, dft_shift):
        mask = np.ones((dft_shift.shape[0], dft_shift.shape[1], 2), np.float32)
        centerX, centerY = (int(dft_shift.shape[0] / 2), int(dft_shift.shape[1] / 2))
        mask[centerX -self.r:centerX + self.r, centerY - self.r:centerY + self.r, :] = 0
        dft_shift = dft_shift*mask

        return dft_shift

class RotationFilter(Filter):
    def __init__(self, angle):
        self.angle = angle

    def processing(self, dft_shift):
        centerX, centerY = (int(dft_shift.shape[0] / 2) , int(dft_shift.shape[1] / 2))
        rot_matrix=cv.getRotationMatrix2D((centerX,centerY),self.angle,1.0)
        dft_shift = cv.warpAffine(dft_shift, rot_matrix, (dft_shift.shape[1], dft_shift.shape[0]), flags=cv.INTER_LINEAR,
                              borderMode=cv.BORDER_CONSTANT)

        return dft_shift

class DegradateFilter(Filter):
    def __init__(self):
        pass

    def processing(self, dft_shift):
        mask = np.random.random((dft_shift.shape[0], dft_shift.shape[1],2))* 0.5 + 1.0
        dft_shift = dft_shift * mask

        return dft_shift

class ConvolutionFilter(Filter):
    def __init__(self, mask):
        self.mask = mask

    def processing(self, dft_shift):
        res_mask = np.zeros((dft_shift.shape[0], dft_shift.shape[1]), np.float32)
        res_mask[:self.mask.shape[0], :self.mask.shape[1]] = self.mask
        dft = dft_image.dft.DFT()
        mask = dft.dft(res_mask)

        dft_shift = dft_shift * mask

        return dft_shift