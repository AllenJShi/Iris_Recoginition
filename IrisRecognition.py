from IrisLocalization import *
from IrisNormalization import *
from IrisMatching import *
from ImageEnhancement import *
from FeatureExtraction import *
from PerformanceEvaluation import *
from utils import *

import random

if __name__=="__main__":
    subject = random.randint(0,107)
    print(subject)
    train = readDataset()   
    img = train[0][subject]
    # img = train[0][1]
    # 81 55 10 94
    # print(img.shape)
    inner_circle, outer_circle = irisLocalization(img)
    img_norm = irisNormalization(img,inner_circle,outer_circle)
    img_enhance = imageEnhancement(img_norm)
    plt.imshow(img_enhance)
    plt.show()
    
    feature_vect = featureExtraction(img_enhance)
    print(len(feature_vect))
