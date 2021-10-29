from IrisLocalization import *
# from IrisNormalization import *
from IrisMatching import *
from FeatureExtraction import *
from PerformanceEvaluation import *
from utils import *


if __name__=="__main__":
    train = readDataset()
    img = train[0][23]
    print(img.shape)
    irisLocalization(img)