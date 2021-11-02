from IrisLocalization import *
from IrisNormalization import *
from IrisMatching import *
from ImageEnhancement import *
from FeatureExtraction import *
from PerformanceEvaluation import *
from utils import *

import random

if __name__=="__main__":
    subject = random.randint(0,108*3-1)
    print(subject)
    # train,test= readDataset(train=True,test=True)   
    # np.save("train",train)
    # np.save("test",test)
    
    
    train = np.load("train.npy")
    test = np.load("test.npy")
    
    img = train[subject]
    # img = train[71]
    # 81 55 10 94
    # print(img.shape)
    inner_circle, outer_circle = irisLocalization(img)
    img_norm = irisNormalization(img,inner_circle,outer_circle)
    img_enhance = imageEnhancement(img_norm)
    plt.imshow(img_enhance)
    plt.show()
    
    feature_vect = featureExtraction(img_enhance)
    # print((feature_vect))
