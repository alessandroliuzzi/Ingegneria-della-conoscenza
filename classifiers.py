from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier



def knnClassifier(X_train, Y_train):
    print("Building KNN Classifier:")
    classifier = KNeighborsClassifier(n_neighbors=4, algorithm='auto', p=2, metric='minkowski', leaf_size=5)
    classifier.fit(X_train, Y_train)
    return classifier


def bayesianClassifier(X_train, Y_train):
    print("Building Bayesian Classifier:")
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    return classifier


def extraTreesClassifier(X_train, Y_train):
    print("Building ExtraTrees Classifier:")
    classifier = ExtraTreesClassifier(n_estimators=275, max_depth=25, min_samples_leaf=1, min_samples_split=2, criterion="gini",
                                      random_state=8, n_jobs=4, bootstrap='false', max_features=20)
    classifier.fit(X_train, Y_train)
    return classifier


def randomForestClassifier(X_train, Y_train):
    print("Building Random Forest Classifier:")
    classifier = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=15, 
                                   max_samples=45,
                                   min_samples_leaf=1,
                                   criterion='gini', min_samples_split=2,
                                   random_state=8,
                                   n_jobs=4)

    classifier.fit(X_train, Y_train)
    return classifier



def getPred(classifier, test):
    
    return classifier.predict(test)[0], max(classifier.predict_proba(test)[0])
