import cv2
import numpy as np

def imageEnhancement(img):
    return cv2.equalizeHist(img.astype(np.uint8))