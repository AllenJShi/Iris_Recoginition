import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import imag

def irisLocalization(img):
    """
        1.Project the image in the vertical and horizontal
        direction to approximately estimate the center
        coordinates ðXp; YpÞ of the pupil. Since the pupil is
        generally darker than its surroundings, the coordinates
        corresponding to the minima of the two
        projection profiles are considered as the center
        coordinates of the pupil.
        2. Binarize a 120  120 region centered at the point
        ðXp; YpÞ by adaptively selecting a reasonable threshold
        using the gray-level histogram of this region. The
        centroid of the resulting binary region is considered
        as a more accurate estimate of the pupil coordinates.
        In this binary region, we can also roughly compute
        the radius of the pupil
        3. Calculate the exact parameters of these two circles
        using edge detection (Canny operator in experiments)
        and Hough transform in a certain region
        determined by the center of the pupil.
        
        In experiments, we perform the second step twice 
        for a reasonably accurate estimate. 
    Args:
        img ([type]): This will reduce the region for edge
        detection and the search space of Hough transform and,
        thus, result in lower computational demands.
    """
    
    # Project the image in the vertical and horizontal
    # direction to approximately estimate the center
    # coordinates Xp; Yp of the pupil
    xp, yp = getPupilCentroid(img)

    # 2. Binarize a 120 X 120 region centered at the point
    # Xp; Yp by adaptively selecting a reasonable threshold
    # using the gray-level histogram of this region. 
    region_ = img[xp-60:xp+60,yp-60:yp+60]
    _, img_binary = cv2.threshold(region_,100,255,cv2.THRESH_BINARY)

    plt.imshow(img_binary,cmap="gray")
    plt.show()
    
    # The centroid of the resulting binary region is considered
    # as a more accurate estimate of the pupil coordinates.
    vertical_projection = projectImg(img=img_binary,axis=0)
    horizontal_projection = projectImg(img=img_binary,axis=1)
    
    xp = np.argmin(vertical_projection) + 60
    yp = np.argmin(horizontal_projection) + 60

    plt.imshow(region_,cmap="gray")
    plt.show()
    
    # In this binary region, we can also roughly compute
    # the radius of the pupil
    radius = 0 
    

def projectImg(img,axis):
    return np.sum(img, axis=axis)


def getPupilCentroid(img):
    vertical_projection = projectImg(img=img,axis=0)
    horizontal_projection = projectImg(img=img,axis=1)
    
    xp = np.argmin(vertical_projection)
    yp = np.argmin(horizontal_projection)
    
    return xp, yp
    

def radiusCalc(size,centroid):
    return 0.5*(size - centroid)
    
    
    
    