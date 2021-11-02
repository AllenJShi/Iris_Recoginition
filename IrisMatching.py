from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid

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
    crr = CRR(y_test,yhat)
    return crr, clf.centroids_
    
def CRR(y_test,yhat):
    return (y_test==yhat).sum()/len(y_test)