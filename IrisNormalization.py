import numpy as np
from math import pi,sin,cos
import math

"""Refer to [https://journals.sagepub.com/doi/pdf/10.1177/1729881417703931]
"""

def irisNormalization(img:np.ndarray,inner_circle:np.ndarray,outer_circle:np.ndarray,offset=0):
    '''
    Project the original iris from a Cartesian coordinate system into a doubly dimensionless pseudopolar coordinate.
    Counterclockwise unwrap the iris ring to a rectangular block, and normalizes irises of different size to the same size.

    Args:
        img: image to be processed
        inner_circle: detected inner_circle parameters from localization
        outer_circle: detected outer_circle parameters from localization
        offset: offset determines the starting point while unwrapping the iris ring

    Returns:
        img_normalized: a normalized iris image
    '''

    M,N = 64, 512
    # create a placeholder for normalized image
    img_normalized = np.zeros((M,N))

    try:
        [xp_,yp_,rp] = inner_circle.astype(int).flatten()
        [xi_,yi_,ri] = outer_circle.astype(int).flatten()
    except:
        [xp_,yp_,rp] = inner_circle.astype(int).flatten()[:3]
        [xi_,yi_,ri] = outer_circle.astype(int).flatten()[:3]

    # this for-loop reassign each pixel value to the 
    # normalized image by converting each corresponding 
    # original pixel into the new coordinate system
    for X in range(N):
        for Y in range(M):
            theta = 2*pi*(X/N) + offset
            # if we use rotation and the angle turn to be greater
            # than 2pi, then we need to find the equivalent theta
            if theta > 2*pi:
                theta -= 2*pi
            xp,yp = unwrap(xp_,yp_,rp,theta)
            xi,yi = unwrap(xi_,yi_,ri,theta)

            x = int(xp + (xi - xp)*(Y/M))
            y = int(yp + (yi - yp)*(Y/M))
            try:
                img_normalized[Y,X] = img[y,x]
            except:
                continue

    return img_normalized


def unwrap(x,y,r,theta):
    """resolve the polor to cartisean coordinate,
    unfold the iris ring coordinates into rect (x,y)

    Args:
        x : x-coordinate in original image
        y : y-coordinate in original image
        r : radius in original image
        theta : angle in original image

    Returns:
        tuple of new coordinates: new coordinates for normalization
    """
    # this method normalizes irises of different size to the same size.
    # Similar to this scheme, we counterclockwise unwrap the iris
    # ring to a rectangular block with a fixed size
    x += r*cos(theta)
    y += r*sin(theta)
    return x,y