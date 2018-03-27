from data_prep import *
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold as VT
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

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
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(x)
    for k in ks:
        kmeans = KMeans(n_clusters=k)#, random_state=20, n_init=20)
        kmeans.fit(X)
        ss_km.append(metrics.silhouette_score(X, kmeans.labels_))
        ars_km.append(metrics.adjusted_rand_score(y, kmeans.labels_))
        vms_km.append(metrics.v_measure_score(y, kmeans.labels_))
        em = GaussianMixture(n_components=k)#, random_state=20, n_init=20)
        em.fit(x)
        pred_labels_em = em.predict(x)
        ss_em.append(metrics.silhouette_score(x, pred_labels_em))
        ars_em.append(metrics.adjusted_rand_score(y, pred_labels_em))
        vms_em.append(metrics.v_measure_score(y, pred_labels_em))
    d = {'k value' : ks,
         'Kmeans_silh' : ss_km,
         'EM_silh' : ss_em,
         'Kmeans_vmeas' : vms_km,
         'EM_vmeas' : vms_em,
         'Kmeans_adj_rand' : ars_km,
         'EM_adj_rand' : ars_em}
    df = pd.DataFrame(d)
    df.set_index('k value', inplace=True)
    #print(df)
    styles = ['rs-', 'ro-', 'r^-', 'bs-', 'bo-', 'b^-']
    fontP = FontProperties()
    fontP.set_size('small')
    filename = dataset+"_clustering_tests"
    plt.figure()
    df.plot(title='Kmeans_EM Comparisons for '+dataset, style=styles)
    plt.legend(loc='best', prop=fontP)
    plt.savefig(filename)

def dimension_reduction(dataset, x, y, ns):
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(x)
    rec_errs_pca = []
    rec_errs_ica = []
    rec_errs_rp = []
    rec_errs = []
    for n in ns:
        pca = PCA(n_components=n)
        reduced_X = pca.fit_transform(X)
#        reconstructed_X = scaler.inverse_transform(pca.inverse_transform(reduced_X))  
#        error = sum(map(np.linalg.norm, reconstructed_X-X))
        reconstructed_X = pca.inverse_transform(reduced_X)
        error = np.linalg.norm((X-reconstructed_X), None)
        rec_errs_pca.append(error)

        ica = FastICA(n_components=n)
        reduced_X = ica.fit_transform(X)
#        reconstructed_X = scaler.inverse_transform(ica.inverse_transform(reduced_X))
#        error = sum(map(np.linalg.norm, reconstructed_X - X))        
        reconstructed_X = ica.inverse_transform(reduced_X)
        error = np.linalg.norm((X-reconstructed_X), None)
        rec_errs_ica.append(error)

        grp = GaussianRandomProjection(n_components=n)
        reduced_X = grp.fit_transform(X)
        pinv = np.linalg.pinv(grp.components_)
#        reconstructed_X = scaler.inverse_transform(np.dot(reduced_X, pinv.T))  
#        error = sum(map(np.linalg.norm, reconstructed_X-X))
        reconstructed_X = np.dot(reduced_X, pinv.T)
        error = np.linalg.norm((X-reconstructed_X), None)
        rec_errs_rp.append(error)

        skb = SelectKBest(mutual_info_classif, k=n)
        reduced_X = skb.fit_transform(X, y)
        reconstructed_X = skb.inverse_transform(reduced_X)
        error = np.linalg.norm((X-reconstructed_X), None)
        rec_errs.append(error)
        #print('X: {0}\treduced X: {1}'.format(X[0],reduced_X[0]))
    d = {'number of components to keep' : ns,
         'PCA' : rec_errs_pca,
         'ICA' : rec_errs_ica,
         'Randome Projection' : rec_errs_rp,
         'SelectKBest: mutual_info_classif' : rec_errs}
    #print(d)
    df = pd.DataFrame(d)
    df.set_index('number of components to keep', inplace=True)
    #print(df)
    styles = ['s-', 'o-', '^-', 'p-']
    fontP = FontProperties()
    fontP.set_size('small')
    filename = dataset+"_dim_red_tests"
    plt.figure()
    ax = df.plot(title='Dimensionality Reduction Algorithms Comparisons for '+dataset, style=styles)
    ax.set_ylabel("Reconstruction Error")
    plt.legend(loc='best', prop=fontP)
    plt.savefig(filename)

if __name__=='__main__':
    np.random.seed(42)
    dataset1 = 'car.data'
    print("-----------------------------------Dataset 1--------------------------------------")
    x, y = load_data(dataset1)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.3)
    dataset1 = dataset1.replace('.', '_')
    
    print("--------------------------Experiment 1------------------------------")
    ks = [2,3,4,5,6,7]
#    clustering(dataset1, x, y, ks)

    print("--------------------------Experiment 2------------------------------")
    ns = [1,2,3,4,5]
    dimension_reduction(dataset1, x, y, ns)

    dataset2 = 'tic-tac-toe.data'
    print("-----------------------------------Dataset 2--------------------------------------")
    x, y = load_data2(dataset2,attributes=10)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.2)
    dataset2 = dataset2.replace('.', '_')

    print("--------------------------Experiment 1------------------------------")
    ks = [2,3,4,5]
#    clustering(dataset2, x, y, ks)

    print("--------------------------Experiment 2------------------------------")
    ns = [1,2,3,4,5,6,7,8]
    dimension_reduction(dataset2, x, y, ns)

