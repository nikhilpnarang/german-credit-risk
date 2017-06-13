import model
import preprocessing
from defines import Metadata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def main():
    metadata = Metadata()
    data, labels = preprocessing.load(metadata)
    data = preprocessing.encode(data, metadata.COLUMNS)

    # divide data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2) #, random_state=33)

    # run classifiers classifiers
    clf_base = model.baseline_classifier(x_train, y_train)
    clf_nb = model.naive_bayes_classifier(x_train, y_train, metadata.COLUMNS)
    clf_knn = model.knn_classifier(x_train, y_train, metadata.COLUMNS)
    clf_svm = model.svm_classifier(x_train, y_train, metadata.COLUMNS)

    # filter best classifier
    clf = [(clf[1].best_score_, clf) for clf in [('base', clf_base),
                                                 ('knn', clf_knn),
                                                 ('svm', clf_svm),
                                                 ('nb', clf_nb)]]
    name, clf = max(clf, key=lambda x: x[0])[1]

    # predict test set
    y_pred = clf.predict(x_test)
    print 'Best classifier: %s' % name
    print '\taccuracy: %0.3f\n' % accuracy_score(y_test, y_pred)
    print classification_report(y_test, y_pred)

if __name__ == '__main__':
    main()
