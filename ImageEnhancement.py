import cv2
import numpy as np
import matplotlib.pyplot as plt
"""In order to obtain a more
well-distributed texture image, we first approximate intensity variations 
across the whole image. The mean of each
16 X 16 small block constitutes a coarse estimate of the
background illumination.
"""

# def imageEnhancement(img):
#     return cv2.equalizeHist(img.astype(np.uint8))


def imageEnhancement(img):
    cimg = img.copy()
    background = meanFilter(cimg)
    removed = removeBackground(img,background)
    img_hist = enhanceIllumination(removed).astype(np.uint8)
    # return cv2.equalizeHist(removed.astype(np.uint8))
    return img_hist


def meanFilter(img:np.ndarray,size=16):
    nrow,ncol = tuple(map(lambda x: int(x/size), img.shape))
    # print(nrow,ncol)
    background = np.zeros((nrow,ncol))
    for row in range(nrow):
        for col in range(ncol):
            value = np.mean(img[row*size: (row+1) * size,col*size: (col +1) * size])
            background[row,col] = value
    
    background = cv2.resize(background,None,fx=size, fy=size, interpolation = cv2.INTER_CUBIC)
    _,ax = plt.subplots(1,2)
    ax[0].imshow(background)
    ax[1].imshow(img)
    plt.show()
    # print(background.shape)
    return background


def enhanceIllumination(img,size=32):
    nrow,ncol = tuple(map(lambda x: int(x/size), img.shape))
    img_hist = np.zeros(img.shape)
    for row in range(nrow):
        for col in range(ncol):
            img_hist[row*size: (row+1) * size,col*size: (col +1) * size] \
                = equalizeHelper(img[row*size: (row+1) * size,col*size: (col +1) * size])
    return img_hist


def removeBackground(img,background):
    return img-background

def equalizeHelper(img):
    return cv2.equalizeHist(np.array(img,dtype=np.uint8))
    


if __name__=="__main__":
    meanFilter(np.array([[1,2]]))