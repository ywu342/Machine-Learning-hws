from data_prep import *
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture

def plot_curves(title, df, xlabel, ylabel, styles, filename, flag=False, dotline=0, line_label='', show100=False):
    colors = plt.cm.rainbow(np.linspace(1, 0, len(Y)))
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (flag):
        plt.xticks(x, x)
    if (dotline!=0):
        tmp = np.zeros(len(x))
        tmp[:] = dotline
        plt.plot(x, tmp, color='black', label=line_label, lw=0.7, ls='dashed')
        tmp100 = np.zeros(len(x))
        tmp100[:] = 100
        if (show100):
            plt.plot(x, tmp100, color='black', label='train accuracy of raw data', lw=0.7, ls='dotted')
#    for (y, label, c) in zip(y, curve_labels, colors):
#        plt.plot(x, y, color=c, label=label, lw=2.0)
    df.plot()
    plt.legend(loc='best')
    plt.savefig(filename)
    return plt

def clustering(dataset, x, y, ks):
    ars_km = []
    ss_km = []
    vms_km = []
    ars_em = []
    ss_em = []
    vms_em = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=10)
        kmeans.fit(x)
        ss_km.append(metrics.silhouette_score(x, kmeans.labels_))
        ars_km.append(metrics.adjusted_rand_score(y, kmeans.labels_))
        vms_km.append(metrics.v_measure_score(y, kmeans.labels_))
        em = GaussianMixture(n_components=k, n_init=10, random_state=10)
        em.fit(x)
        pred_labels_em = em.predict(x)
        ss_em.append(metrics.silhouette_score(x, pred_labels_em))
        ars_em.append(metrics.adjusted_rand_score(y, pred_labels_em))
        vms_em.append(metrics.v_measure_score(y, pred_labels_em))
    d = {'k value' : ks,
         'Kmeans_silh' : ss_km,
         'Kmeans_vmeas' : vms_km,
         'Kmeans_adj_rand' : ars_km,
         'EM_silh' : ss_em,
         'EM_vmeas' : vms_em,
         'EM_adj_rand' : ars_em}
    df = pd.DataFrame(d)
    df.set_index('k value', inplace=True)
    styles = ['rs-', 'ro-', 'r^-', 'bs-', 'bo-', 'b^-']
    filename = dataset+"_clustering_tests"
    plt.figure()
    df.plot(title='Kmeans_EM Comparisons for '+dataset, style=styles)
    plt.legend(loc='best')
    plt.savefig(filename)

if __name__=='__main__':
    np.random.seed(42)
    dataset1 = 'car.data'
    print("-----------------------------------Dataset 1--------------------------------------")
    x, y = load_data(dataset1)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.3)
    ks = [2,3,4,5,6,7]
    dataset1 = dataset1.replace('.', '_')
    clustering(dataset1, x, y, ks)


    dataset2 = 'tic-tac-toe.data'
    print("-----------------------------------Dataset 2--------------------------------------")
    x, y = load_data2(dataset2,attributes=10)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.2)
    ks = [2,3,4,5]
    dataset2 = dataset2.replace('.', '_')
    clustering(dataset2, x, y, ks)
