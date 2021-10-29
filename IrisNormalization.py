import numpy as np
from math import pi

"""Refer to [https://journals.sagepub.com/doi/pdf/10.1177/1729881417703931]
"""

def irisNormalization(img):
    M,N = 64, 512
    # create a placeholder for normalized image
    img_normalized = np.zeros((M,N))
    

    
    for X in range(M):
        for Y in range(N):
            theta = 2*pi*(X/N)
            
            rp = 
            ri = 
            
            xp = 
            yp = 
            
            xi = 
            yi = 
            
            x = xp + ((xi - xp))*(Y/M)
            y = yp + ((yi - yp))*(Y/M)
            img_normalized[X,Y] = img[x,y]

    return img_normalized
