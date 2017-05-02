import numpy
from defines import Types
from scaler import NumericScaler
from binomial import BinomialClassifier
from sklearn import feature_selection
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix


def baseline_classifier(data, labels):
    print 'Applying baseline classification'
    clf = BinomialClassifier(0.7)
    pred = cross_val_predict(clf, data, labels, cv=10)
    clf.best_score_ = accuracy_score(labels, pred)

    print 'accuracy: %0.3f' % accuracy_score(labels, pred)
    print '\n',
    print classification_report(labels, pred)

    return clf

def knn_classifier(data, labels, columns):
    print 'Applying k-nearest neighbor classification'
    # create param grid
    n_numeric = len([c.TYPE for c in columns if c.TYPE is Types.NUMERICAL and c.CATEGORIES is None])
    n_neighbors = list(range(1, 51, 1))
    parameters = dict(knn__n_neighbors=n_neighbors)

    # create model pipeline
    ns = NumericScaler(n_numeric)
    rf = RandomForestClassifier(random_state=8)
    knn = KNeighborsClassifier()
    rfe = feature_selection.RFE(rf)
    pipe = Pipeline(steps=[('ns', ns),
                           ('rfe', rfe),
                           ('knn', knn)])

    # run grid search with 10-fold cross validation
    clf = GridSearchCV(pipe, parameters, cv=10, verbose=1)
    clf.fit(data, labels)
    pred = clf.predict(data)

    print 'accuracy: %0.3f' % clf.best_score_
    print 'Best parameters set: '
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    print '\n',
    print classification_report(labels, pred)

    return clf

def svm_classifier(data, labels, columns):
    print 'Applying SVM classification with RBF kernel'
    # create param grid
    n_numeric = len([c.TYPE for c in columns if c.TYPE is Types.NUMERICAL and c.CATEGORIES is None])
    C = [0.1, 1, 10, 100, 1000]
    gamma = ['auto', 1, 0.1, 0.001, 0.0001]
    parameters = dict(svm__C=C,
                      svm__gamma=gamma)

    # create model pipeline
    ns = NumericScaler(n_numeric)
    rf = RandomForestClassifier(random_state=2)
    rfe = feature_selection.RFE(rf)
    svm = SVC(kernel='rbf', random_state=17)
    pipe = Pipeline(steps=[('ns', ns),
                           ('rfe', rfe),
                           ('svm', svm)])

    # run grid search with 10-fold validation
    clf = GridSearchCV(pipe, parameters, cv=10, verbose=1)
    clf.fit(data, labels)
    pred = clf.predict(data)

    print 'accuracy: %0.3f' % clf.best_score_
    print 'Best parameters set: '
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    print '\n',
    print classification_report(labels, pred)

    return clf

def naive_bayes_classifier(data, labels, columns):
    print 'Applying Naive Bayes classification'
    # create param grid
    n_numeric = len([c.TYPE for c in columns if c.TYPE is Types.NUMERICAL and c.CATEGORIES is None])
    n_components = list(range(1, data.shape[1] + 1, 1))
    parameters = dict(pca__n_components=n_components)

    # create model pipeline
    ns = NumericScaler(n_numeric, with_std=False)
    rf = RandomForestClassifier(random_state=2)
    rfe = feature_selection.RFE(rf)
    pca = decomposition.PCA()
    gnb = GaussianNB()
    pipe = Pipeline(steps=[('ns', ns),
                           ('pca', pca),
                           ('gnb', gnb)])

    # run grid search with 10-fold validation
    clf = GridSearchCV(pipe, parameters, cv=10, verbose=1)
    clf.fit(data, labels)
    pred = clf.predict(data)

    print 'accuracy: %0.3f' % clf.best_score_
    print 'Best parameters set: '
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    print '\n',
    print classification_report(labels, pred)

    return clf
