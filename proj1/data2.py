from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from model_select import *

if __name__=='__main__':
    ### second dataset
    dataset2 = 'tic-tac-toe.data'
    print("-----------------------------------Dataset 2--------------------------------------")
    x, y = load_data2(dataset2,attributes=10)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.2)
#    print('x_train: {} \ny_train: {} \nx_test: {} \ny_test: {}'.format(x_train, y_train, x_test, y_test))

    nn_validation(dataset2, x_train, y_train, x_test, y_test)
    svm_validation(dataset2, x_train, y_train, x_test, y_test)
    dt_validation(dataset2, x_train, y_train, x_test, y_test)
    knn_validation(dataset2, x_train, y_train, x_test, y_test)
    boost_validation(dataset2, x_train, y_train, x_test, y_test, minss=2, maxd=1)

    models = []
    models.append(('SVM', SVC(gamma=0.1, kernel='rbf')))
    models.append(('NeuralNetwork', MLPClassifier(hidden_layer_sizes=51, activation='relu')))
    models.append(('DecisionTree', DecisionTreeClassifier(max_depth=1, min_samples_split=2)))
    models.append(('Boost', AdaBoostClassifier(n_estimators=61, learning_rate=1.0, base_estimator=DecisionTreeClassifier(min_samples_split=2, max_depth=1))))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=9, weights='uniform')))
    compare_algorithms_precision(dataset2, models, x_train, y_train, x_test, y_test)
    compare_algorithms(dataset2, models, x_train, y_train, x_test, y_test)
    for name, model in models:
        generate_learning_curve(dataset2, name, model, x_train, y_train)
