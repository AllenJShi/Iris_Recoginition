import numpy as np
from math import pi,sin,cos,floor

"""Refer to [https://journals.sagepub.com/doi/pdf/10.1177/1729881417703931]
"""

def irisNormalization(img:np.ndarray,inner_circle:np.ndarray, outer_circle:np.ndarray):
    M,N = 64, 512
    # create a placeholder for normalized image
    img_normalized = np.zeros((M,N))
    
    [xp_,yp_,rp] = inner_circle.astype(int).flatten()
    [xi_,yi_,ri] = outer_circle.astype(int).flatten()
    
    for Y in range(M):
        for X in range(N):
            theta = 2*pi*(X/N)
            xp,yp = unwrap(xp_,yp_,rp,theta)
            xi,yi = unwrap(xi_,yi_,ri,theta)

            x = int(xp + (xi - xp)*(Y/M))
            y = int(yp + (yi - yp)*(Y/M))
  
            img_normalized[Y,X] = img[y,x]

    return img_normalized


def unwrap(x,y,r,theta):
    # this method normalizes irises of different size to the same size.
    # Similar to this scheme, we counterclockwise unwrap the iris
    # ring to a rectangular block with a fixed size
    x += int(r*cos(theta))
    y += int(r*sin(theta))
    return x,y