import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, inf
from scipy.spatial.distance import euclidean

def irisLocalization(img):
    """
        1.Project the image in the vertical and horizontal
        direction to approximately estimate the center
        coordinates (Xp; Yp) of the pupil. Since the pupil is
        generally darker than its surroundings, the coordinates
        corresponding to the minima of the two
        projection profiles are considered as the center
        coordinates of the pupil.
        2. Binarize a 120 * 120 region centered at the point
        Xp; Yp by adaptively selecting a reasonable threshold
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
        
    Return:
        pupil and iris circles after localization
    """
    
    # Project the image in the vertical and horizontal
    # direction to approximately estimate the center
    # coordinates Xp; Yp of the pupil
    xp, yp = getPupilCentroid(img)
    # print(xp,yp)
    
    # 2. Binarize a 120 X 120 region centered at the point
    # Xp; Yp by adaptively selecting a reasonable threshold
    # >>>>>> using the gray-level histogram of this region. 
    # The centroid of the resulting binary region is considered
    # as a more accurate estimate of the pupil coordinates.
    xp, yp = adaptiveCentroid(img,xp,yp,size=60)

    # In this binary region, we can also roughly compute
    # the radius of the pupil
    cimg = img.copy()
    x0, x1,y0, y1  = getCorner(xp, yp, half_width=60)
    try:
        # region = cv2.cvtColor(img[x0:x1,y0:y1],cv2.COLOR_GRAY2BGR)
        _, region = cv2.threshold(img[y0:y1,x0:x1], 60, 255, cv2.THRESH_BINARY)
    except:
        xp = int(img.shape[0]/2)
        yp = int(img.shape[1]/2)
        # region = cv2.cvtColor(img[yp-60:yp+60,xp-60:xp+60],cv2.COLOR_GRAY2BGR)
        _, region = cv2.threshold(img[yp-60:yp+60,xp-60:xp+60], 60, 255, cv2.THRESH_BINARY)
    
    # Calculate the exact parameters of these two circles
    # using edge detection (Canny operator in experiments) 
    # and Hough transform in a certain region
    # determined by the center of the pupil.
    
    edges = cv2.Canny(region,100,200)
    minR = 0
    maxR = 0
    inner_circle = cv2.HoughCircles(edges,
                                    cv2.HOUGH_GRADIENT, 
                                    1, 250,
                                    param1=30, param2=10,
                                    minRadius=minR, maxRadius=maxR)
    try:
        inner_circle = np.uint16(np.around(inner_circle))
    except:
        try:
            print(x0,x1,y0,y1)
            region = cv2.cvtColor(img[yp-60:yp+60,xp-60:xp+60],cv2.COLOR_GRAY2BGR)
        except:
            xp = int(img.shape[0]/2)
            yp = int(img.shape[1]/2)
            region = cv2.cvtColor(img[yp-60:yp+60,xp-60:xp+60],cv2.COLOR_GRAY2BGR)
        edges = cv2.Canny(region,100,200)
        inner_circle = cv2.HoughCircles(edges,
                                        cv2.HOUGH_GRADIENT, 
                                        1, 250,
                                        param1=30, param2=10,
                                        minRadius=minR, maxRadius=maxR)

    
    for i in inner_circle[0, :]:
        # draw the outer circle
        i[0] += max(xp-60,0)
        i[1] += max(yp-60,0)
        cv2.circle(cimg, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)


    edges = cv2.Canny(img,20,30)
    minR = 98
    maxR = minR + 20
    outer_circle = cv2.HoughCircles(edges,
                                    cv2.HOUGH_GRADIENT, 
                                    1, 250,
                                    param1=30, param2=10,
                                    minRadius=minR, maxRadius=maxR)

    outer_circle = np.uint16(np.around(outer_circle))

    try:
        flag = (sqrt(((outer_circle.flatten()- \
                    inner_circle.flatten())**2).sum()) \
                > 0.6*outer_circle.flatten()[-1])
    except:
        flag = False
        
    if flag:
        outer_circle[0,0,:2] = inner_circle[0,0,:2]
        outer_circle[0,0,-1] = inner_circle[0,0,-1]+55
    
    for i in outer_circle[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (int(i[0]), int()), 2, (0, 0, 255), 3)

    # plot to view the inner and outer circles
    # plt.imshow(cimg)
    # plt.show()
    
    return inner_circle, outer_circle


def getCorner(x, y, half_width=60):
    """perform sanity check and obtain valid image indice

    Args:
        x (int): estimate center of pupil
        y (int): estimate center of pupil
        half_width (int, optional): the window size. Defaults to 60.

    Returns:
        tuple: coordinates of the restricted area of the pupil
    """
    x0, x1, y0, y1 = x - half_width, x + half_width, y - half_width, y + half_width
    x0 = x0 if x0 >= 0 else 0
    x1 = x1 if x1 <= 280 else 280
    y0 = y0 if y0 >= 0 else 0
    y1 = y1 if y1 <= 320 else 320
    return x0, x1, y0, y1



def projectImg(img,axis):
    """project the image in vertical or horizontal direction

    Args:
        img (array): input image
        axis (int): axis indicator same as numpy

    Returns:
        array: projected image
    """
    try:
        return np.sum(img, axis=axis)
    except:
        pass

def getPupilCentroid(img):
    """
    approximate the center of the pupil

    Args:
        img (array): input image

    Returns:
        tuple: x, y coordinate of the center of pupil
    """
    x_projection = projectImg(img=img,axis=0)
    y_projection = projectImg(img=img,axis=1)
    
    xp = np.argmin(x_projection)
    yp = np.argmin(y_projection)
    
    return xp, yp
    
def adaptiveCentroid(img,xp,yp,size=60):
    """according to the note above, this helps to adaptively find the pupil center

    Args:
        img (array): input image
        xp (int): pupil x-coordinate
        yp (int): pupil y-coordinate
        size (int, optional): window size. Defaults to 60.

    Returns:
        tuple: new coordinates after adjusting twice
    """
    for _ in range(2):
        region_ = img[yp-size:yp+size,xp-size:xp+size].copy()
        _, img_binary = cv2.threshold(region_,64,65,cv2.THRESH_BINARY)
        xp_,yp_ = getPupilCentroid(img_binary)
        xp = xp_ + (xp - size)
        yp = yp_ + (yp - size)
    return xp,yp
        

def radiusCalc(img_binary):
    """binarize image to find a more accurate radius of the circle

    Args:
        img_binary (array): input binary image

    Returns:
        int: radius
    """
    img_binary_ = np.where(img_binary > 1, 0, 1)
    diameter = max(projectImg(img_binary_, axis=0).max(), projectImg(img_binary_, axis=1).max())
    return int(0.5 * diameter)
    