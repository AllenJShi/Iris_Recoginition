from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from ImageEnhancement import *
from IrisNormalization import *
from IrisLocalization import *
from FeatureExtraction import *
from scipy.spatial.distance import euclidean, cosine

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

def generateLabels(N=108,rotate=False,offsets=offsets):
    if rotate:
        y_train = np.repeat(range(N),3*len(offsets))
    else:
        y_train = np.repeat(range(N),3)
    
    y_test = np.repeat(range(N),4)
    
    return y_train, y_test
    


def irisMatching(train, test, n_components = 107,rotate=False, dimReduce = False):
    X_train = []
    X_test = []
    # print(len(train))
    for i,img in enumerate(train):
        X_train_vect = preprocess(img,rotate,offsets)
        # print(len(X_train_vect)) this is a seven-vector 
        X_train.append(X_train_vect)
        print(f"process train image {int((i)/3)}: {int((i))}th")
    
    np.save("X_train",X_train)
    
    for i,img in enumerate(test):
        X_test_vect = preprocess(img,rotate=False,offsets=0)
        X_test.append(X_test_vect)
        print(f"process test image {int((i)/4)}: {int((i))}th")
    
    np.save("X_test",X_test)
    
    y_train, y_test = generateLabels(N=n_components+1,rotate=rotate,offsets=offsets)
    
    # dimension reduction
    if dimReduce:
        X_train_lda, X_test_lda = \
            fisherLinearDiscriminant(X_train,y_train,X_test,n_components=n_components)
        np.save("X_train_lda",X_train_lda)
        np.save("X_test_lda",X_test_lda)
        
        return X_train_lda,y_train, X_test_lda, y_test
    



    # L1 = []
    # L2 = []
    # cosine = []
        
    return X_train,y_train, X_test, y_test



if __name__ == "__main__":
    pass