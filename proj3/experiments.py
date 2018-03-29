from data_prep import *
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(x)
    for k in ks:
        kmeans = KMeans(n_clusters=k)#, random_state=20, n_init=20)
        kmeans.fit(X)
        ss_km.append(metrics.silhouette_score(X, kmeans.labels_))
        ars_km.append(metrics.adjusted_rand_score(y, kmeans.labels_))
        vms_km.append(metrics.v_measure_score(y, kmeans.labels_))

        x = X
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
        reconstructed_X = pca.inverse_transform(reduced_X)
        error = np.linalg.norm((X-reconstructed_X), None)
        rec_errs_pca.append(error)
        #print('pca - reduced X: {0}'.format(reduced_X))

        ica = FastICA(n_components=n)
        ica_reduced_X = ica.fit_transform(X)
        ica_reconstructed_X = ica.inverse_transform(ica_reduced_X)
        ica_error = np.linalg.norm((X-ica_reconstructed_X), None)
        rec_errs_ica.append(ica_error)
        #print('ica - reduced X: {0}'.format(reduced_X))

        grp = GaussianRandomProjection(n_components=n)
        reduced_X = grp.fit_transform(X)
        pinv = np.linalg.pinv(grp.components_)
        reconstructed_X = np.dot(reduced_X, pinv.T)
        error = np.linalg.norm((X-reconstructed_X), None)
        rec_errs_rp.append(error)

        lda = LinearDiscriminantAnalysis(n_components=n)
        reduced_X = lda.fit_transform(X, y)
        pinv = np.linalg.pinv(lda.scalings_[:, 0:n])
        reconstructed_X = np.dot(reduced_X, pinv) + lda.xbar_
        error = np.linalg.norm((X-reconstructed_X), None)
        rec_errs.append(error)
        #print('X: {0}\treduced X: {1}'.format(X[0],reduced_X[0]))
    d = {'Number of components to keep' : ns,
         'PCA' : rec_errs_pca,
         'ICA' : rec_errs_ica,
         'Randomized Projection' : rec_errs_rp,
         'Linear Discriminant Analysis' : rec_errs}
    df = pd.DataFrame(d)
    df.set_index('Number of components to keep', inplace=True)
    styles = ['s-', 'o-', '^-', 'p-']
    fontP = FontProperties()
    fontP.set_size('small')
    filename = dataset+"_dim_red_tests"
    plt.figure()
    ax = df.plot(title='Dimensionality Reduction Algorithms Comparisons for '+dataset, style=styles)
    ax.set_ylabel("Reconstruction error")
    plt.legend(loc='best', prop=fontP)
    plt.savefig(filename)

    eig_vals = np.linalg.eigvals(pca.get_covariance())
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in eig_vals]
    cum_var_exp = np.cumsum(var_exp)
    print('Eigenvalues:')
    print('{}'.format(eig_vals))
    print('Explained variance (%):')
    print('{}'.format(var_exp))
    plt.figure()
    plt.title('PCA generated principal components: all features in '+dataset)
    plt.bar(np.array(range(len(var_exp)))+1, var_exp, color='b', alpha=0.5, align='center',
            label='individual explained variance')
    plt.bar(np.array(range(len(eig_vals)))+1, eig_vals, color='r', alpha=0.3, align='center',
            label='individual eigen value')
    plt.step(np.array(range(len(var_exp)))+1, cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.xlabel('Principal components')
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(loc='best', prop=fontP)
    plt.tight_layout()
    plt.savefig(dataset+"_pca_var_exp")

    ica_kurtos = kurtosis(ica_reduced_X, fisher=False)
    print('ICA kurtosis: {}'.format(ica_kurtos))

#    grp = GaussianRandomProjection(n_components=4)
#    rec_errs_rp = []
#    reduced_X = X
#    for i in range(30):
#        reduced_X = grp.fit_transform(X)
#        if i==9 or i==19 or i==29:
#            pinv = np.linalg.pinv(grp.components_)
#            reconstructed_X = np.dot(reduced_X, pinv.T)
#            error = np.linalg.norm((X-reconstructed_X), None)
#            rec_errs_rp.append(error)
#    plt.figure()
#    plt.title('RP generated principal components: 4 features in '+dataset)
#    plt.bar([10, 20, 30], rec_errs_rp, alpha=0.5, align='center',
#            label='Reconstruction error')
#    plt.xlabel('Number of times RP was run')
#    plt.legend(loc='best')
#    plt.tight_layout()
#    plt.savefig(dataset+"_rp")

def exp3(dataset, n_classes, x, y, ns):
    algorithms = (PCA, FastICA, GaussianRandomProjection, LinearDiscriminantAnalysis)
    ars_pca_km = []
    ars_ica_km = []
    ars_grp_km = []
    ars_lda_km = []
    km_arss = (ars_pca_km, ars_ica_km, ars_grp_km, ars_lda_km)
    ars_pca_em = []
    ars_ica_em = []
    ars_grp_em = []
    ars_lda_em = []
    em_arss = (ars_pca_em, ars_ica_em, ars_grp_em, ars_lda_em)
    km = KMeans(n_clusters=n_classes)#, random_state=10, n_init=10)
    km.fit(x)
    ars_km = [metrics.adjusted_rand_score(y, km.labels_)] * len(ns)
    em = GaussianMixture(n_components=n_classes)
    pred_labels_em = em.fit(x).predict(x)
    ars_em = [metrics.adjusted_rand_score(y, pred_labels_em)] * len(ns)
    for n in ns:
        for (alg, ars) in zip(algorithms, km_arss):
#            km = KMeans(n_clusters=n_classes)
            algorithm = alg(n_components=n)
            reduced_x = algorithm.fit_transform(x, y)
            pred_y = km.fit_predict(reduced_x)
            #ars.append(metrics.adjusted_mutual_info_score(y, pred_y))
            #ars.append(metrics.adjusted_rand_score(y, pred_y))
            ars.append(metrics.v_measure_score(y, pred_y))
    d = {'Number of components to keep' : ns,
         'Unreduced' : ars_km,
         'PCA' : ars_pca_km,
         'ICA' : ars_ica_km,
         'Randomized Projection' : ars_grp_km,
         'Linear Discriminant Analysis' : ars_lda_km}
    df = pd.DataFrame(d)
    df.set_index('Number of components to keep', inplace=True)
    styles = ['p-', 's-', 'o-', '^-','k--']
    fontP = FontProperties()
    fontP.set_size('small')
    filename = dataset+"_exp3_km"
    plt.figure()
    ax = df.plot(title='Kmeans clustering on reduced data for '+dataset, style=styles)
    ax.set_ylabel("V measure")
    plt.legend(loc='best', prop=fontP)
    plt.savefig(filename)

    for n in ns:
        for (alg, ars) in zip(algorithms, em_arss):
    #        em = GaussianMixture(n_components=n_classes)
            algorithm = alg(n_components=n)
            reduced_x = algorithm.fit_transform(x, y)
            em.fit(reduced_x)
            pred_y = em.predict(reduced_x)
            #ars.append(metrics.adjusted_mutual_info_score(y, pred_y))
            #ars.append(metrics.adjusted_rand_score(y, pred_y))
            ars.append(metrics.v_measure_score(y, pred_y))
    d = {'Number of components to keep' : ns,
         'Unreduced' : ars_em,
         'PCA' : ars_pca_em,
         'ICA' : ars_ica_em,
         'Randomized Projection' : ars_grp_em,
         'Linear Discriminant Analysis' : ars_lda_em}
    df = pd.DataFrame(d)
    df.set_index('Number of components to keep', inplace=True)
    styles = ['p-', 's-', 'o-', '^-','k--']
    fontP = FontProperties()
    fontP.set_size('small')
    filename = dataset+"_exp3_em"
    plt.figure()
    ax = df.plot(title='EM clustering on reduced data for '+dataset, style=styles)
    ax.set_ylabel("V measure")
    plt.legend(loc='best', prop=fontP)
    plt.savefig(filename)

if __name__=='__main__':
    np.random.seed(42)
    dataset1 = 'car.data' # 6 attributes 4 classes
    print("-----------------------------------Dataset 1--------------------------------------")
    x, y = load_data(dataset1)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.3)
    dataset1 = dataset1.replace('.', '_')
    
    print("--------------------------Experiment 1------------------------------")
    ks = [2,3,4,5,6,7]
#    clustering(dataset1, x, y, ks)

    print("--------------------------Experiment 2------------------------------")
    ns = [1,2,3,4,5,6]
#    dimension_reduction(dataset1, x, y, ns)

    print("--------------------------Experiment 3------------------------------")
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(x)
    exp3(dataset1, 4, X, y, ns)


    dataset2 = 'tic-tac-toe.data' # 9 attributes 2 classes
    print("-----------------------------------Dataset 2--------------------------------------")
    x, y = load_data2(dataset2,attributes=10)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.2)
    dataset2 = dataset2.replace('.', '_')

    print("--------------------------Experiment 1------------------------------")
    ks = [2,3,4,5]
#    clustering(dataset2, x, y, ks)

    print("--------------------------Experiment 2------------------------------")
    ns = [1,2,3,4,5,6,7,8,9]
#    dimension_reduction(dataset2, x, y, ns)

    print("--------------------------Experiment 3------------------------------")
    X = scaler.fit_transform(x)
    exp3(dataset2, 2, X, y, ns)
