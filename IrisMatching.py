from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from ImageEnhancement import *
from IrisNormalization import *
from IrisLocalization import *
from FeatureExtraction import *
from sklearn.decomposition import PCA
from utils import *


offsets = [-9,-6,-3,0,3,6,9]

def fisherLinearDiscriminant(X_train,y_train,X_test,n_components=107):
    """wrap up the sklearn LDA method

    Args:
        X_train (array): train dataset feature vectors/matrix
        y_train (array): train dataset class labels
        X_test (array): test datasect feature cectors
        n_components (int): number of dimentions asked to retain

    Returns:
        tuple: transformed feature vectors for train and test
    """
    clf = LinearDiscriminantAnalysis(n_components=n_components,solver="svd").fit(X_train,y_train)
    X_train_lda = clf.transform(X_train)
    X_test_lda = clf.transform(X_test)
    return X_train_lda, X_test_lda

def nearestCentroid(X_train,y_train,X_test,y_test,metric):
    """wrap up the sklearn nearest centroid method

    Args:
        X_train (array): train dataset feature vectors/matrix
        y_train (array): train dataset class labels
        X_test (array): test datasect feature cectors
        y_test (array): test dataset class labels
        metric (string): indicators of the metric

    Returns:
        tuple: CRR score and centroids after training
    """
    clf = NearestCentroid(
        metric=metric,shrink_threshold=None
        ).fit(X_train,y_train)
    yhat = clf.predict(X_test)
    crr = CRR(y_test,yhat)
    return crr, clf.centroids_

    
def CRR(y_test,yhat):
    """calculate the CRR score

    Args:
        y_test (array): test dataset labels
        yhat (array): predicted labels

    Returns:
        float: particular CRR measure
    """
    return (y_test==yhat).sum()/len(y_test)


def ROC(results,scores,threshold):
    """calculate the verification scores

    Args:
        results (array): true or false indicator of matching results
        scores (array): min distance measures 
        threshold (float): accept threshold

    Returns:
        tuple: false match rate and false nonmatch rate
    """
    true_accept = ((results == True) & (scores <= threshold)).sum()
    false_accept = ((results == False) & (scores <= threshold)).sum()
    false_reject = ((results == True) & (scores > threshold)).sum()
    true_reject = ((results == False) & (scores > threshold)).sum()
    
    false_match_rate = false_accept/(false_accept+true_reject)
    false_nonmatch_rate = false_reject/(false_reject+true_accept)
    
    return false_match_rate,false_nonmatch_rate


def preprocess(img,rotate=False,offsets=offsets):
    """rotate the img into 7 different angles

    Args:
        img ([type]): img processed after enhancement
        
    """
    feature_vectors = []
    inner_circle, outer_circle = irisLocalization(img)
    
    if rotate:
        # if rotating images, need to rotate by each offset degree
        # this process is slow so it has been replaced by several stages
        # of processing (refer to utils.py)
        for offset in offsets:
            img_norm = irisNormalization(img,inner_circle,outer_circle,offset)
            img_enhance = imageEnhancement(img_norm)
            feature_vector = featureExtraction(img_enhance)
            feature_vectors.append(feature_vector)
    else:
        # otherwise, just process the image directly
        img_norm = irisNormalization(img,inner_circle,outer_circle)
        img_enhance = imageEnhancement(img_norm)
        feature_vector = featureExtraction(img_enhance)
        feature_vectors.append(feature_vector)
    
    return feature_vectors

def generateLabels(N=108,rotate=False,offsets=offsets):
    """generate y labels

    Args:
        N (int, optional): number of classes. Defaults to 108.
        rotate (bool, optional): true if asked to rotate the image. Defaults to False.
        offsets (int, optional): list of degrees to rotate the image. Defaults to offsets.

    Returns:
        tuple: train and test labels
    """
    if rotate:
        y_train = np.repeat(range(N),3*len(offsets))
    else:
        y_train = np.repeat(range(N),3)
    
    y_test = np.repeat(range(N),4)
    
    return y_train, y_test
    


def irisMatching(train, test, n_components = 107,rotate=False, dimReduce = False):
    """generate train and test feature vectors

    Args:
        train (array): raw images in train dataset 
        test (array): raw images in test dataset 
        n_components (int, optional): dimensions if apply reduction. Defaults to 107.
        rotate (bool, optional): true if asked to rotate the image in preprocessing. Defaults to False.
        dimReduce (bool, optional): true if asked to reduce dimension before modeling. Defaults to False.

    Returns:
        tuple: return features vectors and labels for train and test
    """
    X_train = []
    X_test = []
    
    # for i,img in enumerate(train):
    #     X_train_vect = preprocess(img,rotate,offsets)
    #     X_train.append(X_train_vect)
    #     print(f"process train image {int((i)/3)}: {int((i))}th")
    
    # np.save("X_train",X_train)
    
    # for i,img in enumerate(test):
    #     X_test_vect = preprocess(img,rotate=False,offsets=0)
    #     X_test.append(X_test_vect)
    #     print(f"process test image {int((i)/4)}: {int((i))}th")
    
    # np.save("X_test",X_test)
    
    
    if rotate:        
        saveNormalizedImage()
        rotateAll()
        extractFeatureFromRotatedImg()

        X_train = np.load("X_train_rotate.npy")
        # do not rotate test dataset
        X_test = np.load("X_test.npy")
        
    else:
        X_train = np.load("X_train.npy")
        X_test = np.load("X_test.npy")
    
    
    y_train, y_test = generateLabels(N=n_components+1,rotate=rotate,offsets=offsets)


    
    # dimension reduction
    if dimReduce:
        X_train_pca, X_test_pca = \
            principalComponentsAnalysis(X_train,X_test,y_train=y_train)
        np.save("X_train_pca",X_train_pca)
        np.save("X_test_pca",X_test_pca)
        
        return X_train_pca,y_train, X_test_pca, y_test
        
    return X_train,y_train, X_test, y_test


def principalComponentsAnalysis(X_train,X_test,y_train=None):
    """wrap up sklearn PCA method

    Args:
        X_train (array): train dataset feature vectors/matrix
        y_train (array): train dataset class labels
        X_test (array): test datasect feature cectors

    Returns:
        tuple: dimension reduced feature vectors for train and test
    """
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))
    
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    
    pca = PCA().fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca,X_test_pca
