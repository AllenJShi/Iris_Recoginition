from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from ImageEnhancement import *
from IrisNormalization import *
from IrisLocalization import *
from FeatureExtraction import *


offsets = [-9,-6,-3,0,3,6,9]

def fisherLinearDiscriminant(X_train,y_train,X_test,n_components=100):
    clf = LinearDiscriminantAnalysis(n_components).fit(X_train,y_train)
    X_train_lda = clf.transform(X_train)
    X_test_lda = clf.transform(X_test)
    return X_train_lda, X_test_lda

def nearestCentroid(X_train,y_train,X_test,y_test,metric):
    clf = NearestCentroid(
        metric=metric
        ).fit(X_train,y_train)
    yhat = clf.predict(X_test)
    # crr = CRR(y_test,yhat)
    # return crr, clf.centroids_
    return yhat, clf.centroids_
    
def CRR(y_test,yhat):
    return (y_test==yhat).sum()/len(y_test)


def preprocess(img,rotate=False,offsets=offsets):
    """rotate the img into 7 different angles

    Args:
        img ([type]): img processed after enhancement
        
    """
    feature_vectors = []
    inner_circle, outer_circle = irisLocalization(img)
    
    if rotate:
        for offset in offsets:
            img_norm = irisNormalization(img,inner_circle,outer_circle,offset)
            img_enhance = imageEnhancement(img_norm)
            feature_vector = featureExtraction(img_enhance)
            feature_vectors.append(feature_vector)
    else:
        img_norm = irisNormalization(img,inner_circle,outer_circle)
        img_enhance = imageEnhancement(img_norm)
        feature_vector = featureExtraction(img_enhance)
        feature_vectors.append(feature_vector)
    
    return feature_vectors


def irisMatching(train, test, n_components = 107,rotate=False):
    X_train = []
    X_test = []
    for img in train:
        X_train_vect = preprocess(img,rotate,offsets)
        X_train.append(X_train_vect)
        
    for img in test:
        X_test_vect = preprocess(img,rotate=False,offsets=0)
        X_test.append(X_test_vect)
    
    if rotate:
        y_train = np.repeat(range(n_components+1),3*len(offsets))
    else:
        y_train = np.repeat(range(n_components+1),3)
    
    y_test = np.repeat(range(n_components+1),4)
        
    return X_train,y_train, X_test, y_test