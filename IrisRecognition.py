from IrisLocalization import *
from IrisNormalization import *
from IrisMatching import *
from ImageEnhancement import *
from FeatureExtraction import *
from PerformanceEvaluation import *
from utils import *    
    


if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    # read images from dataset and store in npy files
    train,test= readDataset(train=True,test=True)   
    np.save("train",train)
    np.save("test",test)
    
    # load raw image data from npy files
    train = np.load("train.npy")
    test = np.load("test.npy")
    X_train,y_train, X_test, y_test= \
        irisMatching(train, test, n_components = 107,rotate=False, dimReduce = False)
    performanceEvaluation(X_train, y_train, X_test, y_test)
    
    
    
    # ==================================================================================================
    # The following code provides a shortcut by storing and reusing some intermediate image-preprocessing
    # results and thus reduce the overall running time.
    # Caveat: the scripts above must be executed at least once to have 
    # -"train.npy", 
    # -"test.npy",
    # -"X_train.npy",
    # -"X_test.npy"
    # stored in the working directory.
    # Also, be aware that the previous CRR.png, FMR_FNMR.png, "CRR Table.csv" and "ROC Table.csv" will
    # be updated and overwritten for each operation.
    # ==================================================================================================
    # load the preprocessed data directly from npy files
    # X_train = np.load("X_train.npy")
    # X_test = np.load("X_test.npy")
    
    # use rotation to obtain invariants, need to reload "train.npy" to rotate the train dataset
    # train = np.load("train.npy")
    # X_train,y_train, X_test, y_test = irisMatching(train=train, test=None,rotate=True)
    # performanceEvaluation(X_train, y_train, X_test, y_test)
    # CRR result
    # 0.7592592592592593 0.7152777777777778 0.7013888888888888

    # use the stored npy data to save preprocessing time
    # X_train,y_train, X_test, y_test = irisMatching(train=None, test=None,rotate=False)
    # performanceEvaluation(X_train, y_train, X_test, y_test)
    # CRR result
    # 0.8125 0.7708333333333334 0.7662037037037037
    # 0.7870370370370371 0.8009259259259259 0.8541666666666666
    
    
    # use PCA to reduce the dimensionality
    # X_train,y_train, X_test, y_test = irisMatching(train=None, test=None,rotate=False,dimReduce=True)
    # performanceEvaluation(X_train, y_train, X_test, y_test)
    # CRR result
    # 0.8587962962962963 0.7708333333333334 0.7986111111111112
    # 0.006944444444444444 0.011574074074074073 0.004629629629629629
    
    
    
    
    
    



