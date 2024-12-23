import math
import numpy as np

def processing(hist, image):
    av1 = np.array([key*value for key, value in hist.items()]).sum()
    std1 = math.sqrt(np.array([ value * (key - av1)**2 for key, value in hist.items()]).sum())

    av2 = image.mean()
    std2 = image.std()
    image = std1 * (image - av2) / std2 + av1

    return image