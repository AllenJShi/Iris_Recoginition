import cv2
import numpy as np
import matplotlib.pyplot as plt
"""In order to obtain a more
well-distributed texture image, we first approximate intensity variations 
across the whole image. The mean of each
16 X 16 small block constitutes a coarse estimate of the
background illumination.
"""

# straightforward enhancement, this actually yields better results than the method suggested in paper
# def imageEnhancement(img):
#     return cv2.equalizeHist(img.astype(np.uint8))


def imageEnhancement(img):
    """image enhancement by mean firltering and remove illumination in background

    Args:
        img (array): input image

    Returns:
        array: enhanced image
    """
    cimg = img.copy()
    background = meanFilter(cimg)
    removed = removeBackground(img,background)
    img_hist = enhanceIllumination(removed)
    # return cv2.equalizeHist(removed.astype(np.uint8))
    return img_hist


def meanFilter(img:np.ndarray,size=16):
    # Estimated background illumination by bicubic interpolation:
    # The mean of each 16*16 small block constitutes a coarse estimate of the background illumination.
    # This estimate is further expanded to the same size as the normalized image by bicubic interpolation.
    nrow,ncol = tuple(map(lambda x: int(x/size), img.shape))
    background = np.zeros((nrow,ncol))
    for row in range(nrow):
        for col in range(ncol):
            value = np.mean(img[row*size: (row+1) * size,col*size: (col +1) * size],dtype=np.float32)
            background[row,col] = value
    
    background = cv2.resize(background,(img.shape[1],img.shape[0]), interpolation = cv2.INTER_CUBIC)
    
    # view the result
    
    # _,ax = plt.subplots(1,2)
    # ax[0].imshow(background,cmap="gray")
    # ax[1].imshow(img,cmap="gray")
    # plt.show()

    return background


def enhanceIllumination(img,size=32):
    # Enhance the lighting corrected image by histogram equalization in each 32*32 region.
    nrow,ncol = tuple(map(lambda x: int(x/size), img.shape))
    img_hist = np.zeros(img.shape)
    
    # for each 32x32 block, enhance the subimage in place
    for row in range(nrow):
        for col in range(ncol):
            img_hist[row*size: (row+1) * size,col*size: (col +1) * size] \
                = equalizeHelper(img[row*size: (row+1) * size,col*size: (col +1) * size])
    img_hist = cv2.GaussianBlur(img_hist, (3, 3), 0)
    return img_hist.astype(np.uint8)


def removeBackground(img,background):
    # Subtracted the estimated background illumination from the normalized image
    # to compensate for a variety of lighting conditions.
    enhance = img-background
    enhance = enhance - np.amin(enhance.ravel())
    return enhance.astype(np.uint8)  

def equalizeHelper(img):
    # Histogram equalization
    return cv2.equalizeHist(np.array(img,dtype=np.uint8))
    
