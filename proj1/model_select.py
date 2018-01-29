from data_prep import *
import time
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

def compare_algorithms_precision(dataset, models, x_train, y_train, x_test, y_test):
    names = []
    train_errors = []
    test_errors = []
    train_times = []
    test_times = []
    for name, model in models:
        model.fit(x_train, y_train)
        pred_test = model.predict(x_test)
        pred_train= model.predict(x_train)
        train_errors.append(metrics.precision_score(y_train, pred_train, average='macro'))
        test_errors.append(metrics.precision_score(y_test, pred_test, average='macro'))
        train_times.append(metrics.recall_score(y_train, pred_train, average='macro'))
        test_times.append(metrics.recall_score(y_test, pred_test, average='macro'))
        names.append(name)
    ind = np.arange(5)+1
    width=0.3
    fontsize=8
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle(dataset+' Algorithm Comparisons (P/R)')
    ax1.set_ylabel('Precision', color='r')
    bar1_train = ax1.bar(ind-width/2, train_errors, width, color=(1,0,0,0.8), label='Training Precision')
    bar1_test = ax1.bar(ind+width/2, test_errors, width, color=(1,0,0,0.4), label='Testing Precision')
    ax1.legend(loc='best')
    ax2.set_ylabel('Recall', color='b')
    bar2_train = ax2.bar(ind-width/2, train_times, width, color=(0,0,1,0.8), label='Training Recall')
    bar2_test = ax2.bar(ind+width/2, test_times, width, color=(0,0,1,0.4), label='Testing Recall')
    ax2.legend(loc='best')
    bars = [bar1_train, bar1_test, bar2_train, bar2_test]
    i = 0
    axs = [ax1, ax1, ax2, ax2]
    for bar in bars:
        for rect in bar:
            height = rect.get_height()
            axs[i].text(rect.get_x() + rect.get_width()/2., height,
                    '%.5f' % height,
                    ha='center', va='bottom', fontsize=fontsize)
        i+=1
    plt.xticks(ind+width/2)
    ax1.set_xticklabels(names)
    ax2.set_xticklabels(names)
    plt.tight_layout()
    plt.savefig(dataset+"_Algorithms_Comparisons_PR.png", bbox_inches='tight', dpi=500)
#    plt.show()

def model_validation(dataset, model_type, estimator, parameters, x_train, y_train, x_test, y_test):
    #print('x_train: {} \ny_train: {} \nx_test: {} \ny_test: {}'.format(x_train, y_train, x_test, y_test))
    print("### Tuning hyper-parameters for %s" % model_type)
    print()
    clf = GridSearchCV(estimator, parameters, cv=10, n_jobs=-1)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()
#    print("Grid scores on development set:")
#    print()
    means = 100*clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']*100
    x_labels = []
    keys = list(parameters.keys())
    key1 = keys[0]
    key2 = keys[1]
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #print("%0.3f (+/-%0.03f) for %r"
        #      % (mean, std * 2, params))
    for params in clf.cv_results_['params']:
        x_labels.append(str(params[key1])+'/'+str(params[key2]))
    plt.figure()
    plt.title(dataset+": "+model_type+" Model Validation")
    ax = plt.subplot()
    width=4
    fontsize=5
    ax.set_ylabel('Mean Cross Validation Accuracy (%)')
    ax.set_xlabel('Combinations of hyperparameters: '+key1+"/"+key2)
    indices = np.arange(1, len(x_labels)*width*2+1, width*2)
    bar = ax.bar(indices, means, width, alpha =0.6, label='Accuracy', yerr=stds)
    plt.xticks(indices+width/2)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=fontsize)
    ax.yaxis.grid()
    plt.tight_layout()
    for rect in bar:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%.1f' % height,
                ha='center', va='bottom', fontsize=fontsize)
    plt.savefig(dataset+"_"+model_type+".png", bbox_inches='tight', dpi=500)
#    print()
#    plt.scatter(plot_x, plot_y, alpha=0.5)
    #plt.show()
    
    print("Testing Accuracy by the best model: ")
#    y_true, y_pred = y_test, clf.predict(x_test)
#    print(classification_report(y_true, y_pred))
#    print(clf.best_estimator_)
    print(clf.best_estimator_.score(x_test, y_test)*100)
    print()

def svm_validation(dataset, x_train, y_train, x_test, y_test):
    gamma_range = np.logspace(-6, -1, 5)
    parameters = {'gamma':gamma_range,'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
    model_validation(dataset, 'SVM', SVC(), parameters, x_train, y_train, x_test, y_test)

def dt_validation(dataset, x_train, y_train, x_test, y_test):
    max_depth_range = list(np.arange(1, 72, 20))
    max_depth_range.append(None)
    min_split_range = np.arange(2, 10, 2)
    parameters = {'max_depth':max_depth_range, 'min_samples_split':min_split_range} 
    model_validation(dataset, 'Decision Tree', DecisionTreeClassifier(), parameters, x_train, y_train, x_test, y_test)

def boost_validation(dataset, x_train, y_train, x_test, y_test, minss, maxd):
    n_range = np.arange(1, 70, 15)
    lr_range = np.arange(0.1, 1.1, 0.1)
    parameters = {'n_estimators':n_range, 'learning_rate':lr_range} 
    ### pruned base learner
    model_validation(dataset, 'Boosting', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=minss, max_depth=maxd)), parameters, x_train, y_train, x_test, y_test)
    #model_validation(dataset, 'Boosting', AdaBoostClassifier(), parameters, x_train, y_train, x_test, y_test)

def knn_validation(dataset, x_train, y_train, x_test, y_test):
    k_range = np.arange(1, 50, 2)
    parameters = {'n_neighbors':k_range, 'weights':['uniform', 'distance']} 
    model_validation(dataset, 'KNN', KNeighborsClassifier(), parameters, x_train, y_train, x_test, y_test)

def nn_validation(dataset, x_train, y_train, x_test, y_test):
    layer_size_range = np.arange(1, 56, 5)
    #max_iter_range = np.arange(200, 1000, 200)
    hls_range = []
    for i in layer_size_range:
        hls_range.append((i,))
    parameters = {'hidden_layer_sizes':hls_range, 'activation':['relu', 'logistic', 'tanh', 'identity']}#'max_iter':max_iter_range} 
    model_validation(dataset, 'Neural Network', MLPClassifier(), parameters, x_train, y_train, x_test, y_test)

def compare_algorithms(dataset, models, x_train, y_train, x_test, y_test):
    names = []
    train_errors = []
    test_errors = []
    train_times = []
    test_times = []
    for name, model in models:
        trstart_time = time.time()
        model.fit(x_train, y_train)
        train_times.append(time.time()-trstart_time)
        tsstart_time = time.time()
        pred_test = model.predict(x_test)
        test_times.append(time.time()-tsstart_time)
        pred_train= model.predict(x_train)
        train_errors.append(1-metrics.accuracy_score(y_train, pred_train))
        test_errors.append(1-metrics.accuracy_score(y_test, pred_test))
        names.append(name)
    ind = np.arange(5)+1
    width=0.3
    fontsize=8
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle(dataset+' Algorithm Comparisons')
    ax1.set_ylabel('Error', color='r')
    bar1_train = ax1.bar(ind-width/2, train_errors, width, color=(1,0,0,0.8), label='Training Error')
    bar1_test = ax1.bar(ind+width/2, test_errors, width, color=(1,0,0,0.4), label='Testing Error')
    ax1.legend(loc='best')
    ax2.set_ylabel('Time (s)', color='b')
    bar2_train = ax2.bar(ind-width/2, train_times, width, color=(0,0,1,0.8), label='Training Time')
    bar2_test = ax2.bar(ind+width/2, test_times, width, color=(0,0,1,0.4), label='Testing Time')
    ax2.legend(loc='best')
    bars = [bar1_train, bar1_test, bar2_train, bar2_test]
    i = 0
    axs = [ax1, ax1, ax2, ax2]
    for bar in bars:
        for rect in bar:
            height = rect.get_height()
            axs[i].text(rect.get_x() + rect.get_width()/2., height,
                    '%.5f' % height,
                    ha='center', va='bottom', fontsize=fontsize)
        i+=1
    plt.xticks(ind+width/2)
    ax1.set_xticklabels(names)
    ax2.set_xticklabels(names)
    plt.tight_layout()
    plt.savefig(dataset+"_Algorithms_Comparisons.png", bbox_inches='tight', dpi=500)
    #plt.show()

def generate_learning_curve(dataset, model_type, estimator, x_train, y_train, x_test=None, y_test=None, train_sizes=np.linspace(0.1, 1, 10)):
    #print(x_train)
    #print(y_train)
    plt.figure()
    plt.title(dataset+': Learning curve for '+model_type)
    plt.xlabel('Training examples')
    plt.ylabel('Error')
    train_sizes, train_scores, valid_scores = learning_curve( \
        estimator, x_train, y_train, cv=10, n_jobs=-1, train_sizes=train_sizes)
    train_scores = 1-train_scores
    valid_scores = 1-valid_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, label='Training Error', linewidth=3)
    plt.plot(train_sizes, valid_scores_mean, label='Cross Validation Error', linewidth=3)
    plt.legend(loc='best')
    plt.savefig(dataset+"_"+model_type+"_lc.png", bbox_inches='tight', dpi=500)
    #plt.show()

if __name__=='__main__':
    ### first dataset
    dataset1 = 'car.data'
    print("-----------------------------------Dataset 1--------------------------------------")
    x, y = load_data(dataset1)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.3)
    #print('x_train: {} \ny_train: {} \nx_test: {} \ny_test: {}'.format(x_train, y_train, x_test, y_test))

    svm_validation(dataset1, x_train, y_train, x_test, y_test)
    dt_validation(dataset1, x_train, y_train, x_test, y_test)
    knn_validation(dataset1, x_train, y_train, x_test, y_test)
    boost_validation(dataset1, x_train, y_train, x_test, y_test, minss=2, maxd=41)
    nn_validation(dataset1, x_train, y_train, x_test, y_test)

    models = []
    models.append(('SVM', SVC(gamma=0.1, kernel='rbf')))
    models.append(('NeuralNetwork', MLPClassifier(hidden_layer_sizes=51, activation='relu')))#max_iter=520)))
    models.append(('DecisionTree', DecisionTreeClassifier(max_depth=41, min_samples_split=2)))
    models.append(('Boost', AdaBoostClassifier(n_estimators=51, learning_rate=0.8, base_estimator=DecisionTreeClassifier(min_samples_split=2, max_depth=41))))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=7, weights='distance')))
    compare_algorithms_precision(dataset1, models, x_train, y_train, x_test, y_test)
    compare_algorithms(dataset1, models, x_train, y_train, x_test, y_test)
    for name, model in models:
        generate_learning_curve(dataset1, name, model, x_train, y_train)

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
#    boost_validation(dataset2, x_train, y_train, x_test, y_test, minss=2, maxd=None)
    #boost_validation(dataset2, x_train, y_train, x_test, y_test, minss=8, maxd=61)

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

    ### second dataset

#def plot_validation_curve(dataset_name, model_type, param, param_range, cv_scores, train_scores, test_scores):
#    fname = '{}_{}_{}_model_validation.png'.format(model_type, param, dataset_name)
#    cv_scores = 1-cv_scores
#    train_scores = 1-train_scores
#    test_scores = 1-test_scores
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    cv_scores_mean = np.mean(cv_scores, axis=1)
#    cv_scores_std = np.std(cv_scores, axis=1)
#    
#    plt.title("Validation Curve with "+model_type)
#    plt.xlabel(param)
#    plt.ylabel("Error")
##    plt.ylim(0.0, 1.1)
#    lw = 2
#    plt.semilogx(param_range, train_scores_mean, label="Training data Error",
#                 color="darkorange", lw=lw)
#    plt.semilogx(param_range, cv_scores_mean, label="Cross-validation Error",
#                 color="navy", lw=lw)
##    plt.fill_between(param_range, test_scores_mean - test_scores_std,
##                     test_scores_mean + test_scores_std, alpha=0.2,
##                     color="navy", lw=lw)
#    plt.semilogx(param_range, test_scores, label="Test data Error",
#                 color="navy", lw=lw)
#    plt.legend(loc="best")
#    plt.show()
#    plt.savefig(fname)
#
