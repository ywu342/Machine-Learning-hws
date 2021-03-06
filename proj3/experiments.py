from data_prep import *
import time
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
            algorithm = alg(n_components=n)
            reduced_x = algorithm.fit_transform(x, y)
            pred_y = km.fit_predict(reduced_x)
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
            algorithm = alg(n_components=n)
            reduced_x = algorithm.fit_transform(x, y)
            em.fit(reduced_x)
            pred_y = em.predict(reduced_x)
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

def plot_pro_1(x, y, alg, alg_name):
    pca = alg(n_components=2)
    reduced_x = pca.fit_transform(x,y)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(reduced_x)
    h = .02
    x_min, x_max = reduced_x[:, 0].min() - 1, reduced_x[:, 0].max() + 1
    y_min, y_max = reduced_x[:, 1].min() - 1, reduced_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
    ax = plt.subplot(111)
    label_dict = {2:'unacc', 0:'acc', 1:'good', 3:'vgood'}
    for label,marker,color in zip(
        range(0,4),('*', '^', 's', 'o'),('k','blue', 'red', 'green')):

        plt.scatter(x=reduced_x[:,0][y == label],
                y=reduced_x[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    fontP = FontProperties()
    fontP.set_size('small')
    leg = plt.legend(loc='best', prop=fontP)
    leg.get_frame().set_alpha(0.5)
    plt.title(alg_name+': Car projection onto the first 2 principal components')
    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    
    plt.tight_layout
    plt.grid()
    plt.savefig("car_cluster_proj_"+alg_name)

def plot_pro_2(x, y, alg, alg_name):
    pca = alg(n_components=2)
    reduced_x = pca.fit_transform(x,y)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(reduced_x)
    h = .02
    x_min, x_max = reduced_x[:, 0].min() - 1, reduced_x[:, 0].max() + 1
    y_min, y_max = reduced_x[:, 1].min() - 1, reduced_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
    ax = plt.subplot(111)
    label_dict = {1:'positive', 0:'negative'}
    for label,marker,color in zip(
        range(0,2),('*', '^'),('k','blue')):

        plt.scatter(x=reduced_x[:,0][y == label],
                y=reduced_x[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    fontP = FontProperties()
    fontP.set_size('small')
    leg = plt.legend(loc='best', prop=fontP)
    leg.get_frame().set_alpha(0.5)
    plt.title(alg_name+': Tic-Tac-Toe projection onto the first 2 principal components')
    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    
    plt.tight_layout
    plt.grid()
    plt.savefig("ttt_cluster_proj_"+alg_name)

def exp4(dataset, x_train, x_test, y_train, y_test, ns):
    algorithms = (PCA, FastICA, GaussianRandomProjection, LinearDiscriminantAnalysis)
    alg_names = ('PCA', 'ICA', 'Randomized Projection', 'Linear Discriminant Analysis')
    d_train_errors = {'Number of components to keep' : ns,
         'Unreduced' : [],
         'PCA' : [],
         'ICA' : [],
         'Randomized Projection' : [],
         'Linear Discriminant Analysis' : []}
    d_test_errors = {'Number of components to keep' : ns,
         'Unreduced' : [],
         'PCA' : [],
         'ICA' : [],
         'Randomized Projection' : [],
         'Linear Discriminant Analysis' : []}
    d_train_times = {'Number of components to keep' : ns,
         'Unreduced' : [],
         'PCA' : [],
         'ICA' : [],
         'Randomized Projection' : [],
         'Linear Discriminant Analysis' : []}
    d_test_times = {'Number of components to keep' : ns,
         'Unreduced' : [],
         'PCA' : [],
         'ICA' : [],
         'Randomized Projection' : [],
         'Linear Discriminant Analysis' : []}

    nn = MLPClassifier()
    trstart_time = time.time()
    nn.fit(x_train, y_train)
    d_train_times['Unreduced'] = [time.time() - trstart_time] * len(ns)
    tsstart_time = time.time()
    pred_test = nn.predict(x_test)
    d_test_times['Unreduced'] = [time.time()-tsstart_time] * len(ns)
    pred_train= nn.predict(x_train)
    d_train_errors['Unreduced'] = [metrics.accuracy_score(y_train, pred_train)] * len(ns)
    d_test_errors['Unreduced'] =[metrics.accuracy_score(y_test, pred_test)] * len(ns)
    for (alg, name) in zip(algorithms, alg_names):
        for n in ns:
            algorithm = alg(n_components=n)
            reduced_x = algorithm.fit_transform(x_train, y_train)
            reduced_x_tst = algorithm.fit_transform(x_test, y_test)
            nn = MLPClassifier()
            trstart_time = time.time()
            nn.fit(reduced_x, y_train)
            d_train_times[name].append(time.time() - trstart_time)
            tsstart_time = time.time()
            pred_test = nn.predict(reduced_x_tst)
            d_test_times[name].append(time.time()-tsstart_time)
            pred_train= nn.predict(reduced_x)
            d_train_errors[name].append(metrics.accuracy_score(y_train, pred_train))
            d_test_errors[name].append(metrics.accuracy_score(y_test, pred_test))
    df_train_errors = pd.DataFrame(d_train_errors)
    df_train_errors.set_index('Number of components to keep', inplace=True)
    df_test_errors = pd.DataFrame(d_test_errors)
    df_test_errors.set_index('Number of components to keep', inplace=True)
    df_train_times = pd.DataFrame(d_train_times)
    df_train_times.set_index('Number of components to keep', inplace=True)
    df_test_times = pd.DataFrame(d_test_times)
    df_test_times.set_index('Number of components to keep', inplace=True)
    styles = ['p-', 's-', 'o-', '^-','k--']
    filename = dataset+"_exp4"
    plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    df_train_errors.plot(ax=axes[0,0], style=styles)
    axes[0,0].set_ylabel("Training Accuracy")
    df_test_errors.plot(ax=axes[0,1], style=styles)
    axes[0,1].set_ylabel("Testing Accuracy")
    df_train_times.plot(ax=axes[1,0], style=styles)
    axes[1,0].set_ylabel("Training Time")
    df_test_times.plot(ax=axes[1,1], style=styles)
    axes[1,1].set_ylabel("Testing Time")
    axes[0,0].legend_.set_visible(False)
    axes[0,1].legend_.set_visible(False)
    axes[1,1].legend_.set_visible(False)
    axes[1,0].legend_.set_visible(False)
    handles, labels = axes[0,0].get_legend_handles_labels()
    plt.figlegend(handles=handles, labels=labels, loc='upper right', prop=fontP)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.suptitle('NN on Reduced Data for '+dataset)
    plt.savefig(filename)

def exp5(dataset, x_train, x_test, y_train, y_test, c_alg, c_alg_name, n_classes=4, ns=[1,2,3,4,5,6]):
    algorithms = (PCA, FastICA, GaussianRandomProjection, LinearDiscriminantAnalysis)
    alg_names = ('PCA', 'ICA', 'Randomized Projection', 'Linear Discriminant Analysis')
    d_train_errors = {'Number of components to keep' : ns,
         'Unreduced' : [],
         'PCA' : [],
         'ICA' : [],
         'Randomized Projection' : [],
         'Linear Discriminant Analysis' : []}
    d_test_errors = {'Number of components to keep' : ns,
         'Unreduced' : [],
         'PCA' : [],
         'ICA' : [],
         'Randomized Projection' : [],
         'Linear Discriminant Analysis' : []}
    d_train_times = {'Number of components to keep' : ns,
         'Unreduced' : [],
         'PCA' : [],
         'ICA' : [],
         'Randomized Projection' : [],
         'Linear Discriminant Analysis' : []}
    d_test_times = {'Number of components to keep' : ns,
         'Unreduced' : [],
         'PCA' : [],
         'ICA' : [],
         'Randomized Projection' : [],
         'Linear Discriminant Analysis' : []}
    c_algobj = c_alg(n_clusters=n_classes) if c_alg_name!='EM' else c_alg(n_components=n_classes)
    pred_labels_train = c_algobj.fit(x_train).predict(x_train)
    pred_labels_train = np.reshape(pred_labels_train, (len(x_train),1))
    c_x_train = np.hstack((x_train, pred_labels_train))
    pred_labels_test = c_algobj.fit(x_test).predict(x_test)
    pred_labels_test = np.reshape(pred_labels_test, (len(x_test),1))
    c_x_test = np.hstack((x_test, pred_labels_test))
    nn = MLPClassifier()
    trstart_time = time.time()
    nn.fit(c_x_train, y_train)
    d_train_times['Unreduced'] = [time.time() - trstart_time] * len(ns)
    tsstart_time = time.time()
    pred_test = nn.predict(c_x_test)
    d_test_times['Unreduced'] = [time.time()-tsstart_time] * len(ns)
    pred_train= nn.predict(c_x_train)
    d_train_errors['Unreduced'] = [metrics.accuracy_score(y_train, pred_train)] * len(ns)
    d_test_errors['Unreduced'] =[metrics.accuracy_score(y_test, pred_test)] * len(ns)
    for (alg, name) in zip(algorithms, alg_names):
        for n in ns:
            algorithm = alg(n_components=n)
            reduced_X = algorithm.fit_transform(x_train, y_train)
            reduced_X_tst = algorithm.fit_transform(x_test, y_test)
            pred_labels_train = c_algobj.fit(reduced_X).predict(reduced_X)
            pred_labels_train = np.reshape(pred_labels_train, (len(reduced_X),1))
            pred_labels_test = c_algobj.fit(reduced_X_tst).predict(reduced_X_tst)
            pred_labels_test = np.reshape(pred_labels_test, (len(reduced_X_tst),1))
            reduced_x = np.hstack((reduced_X, pred_labels_train))
            reduced_x_tst = np.hstack((reduced_X_tst, pred_labels_test))
            nn = MLPClassifier()
            trstart_time = time.time()
            nn.fit(reduced_x, y_train)
            d_train_times[name].append(time.time() - trstart_time)
            tsstart_time = time.time()
            pred_test = nn.predict(reduced_x_tst)
            d_test_times[name].append(time.time()-tsstart_time)
            pred_train= nn.predict(reduced_x)
            d_train_errors[name].append(metrics.accuracy_score(y_train, pred_train))
            d_test_errors[name].append(metrics.accuracy_score(y_test, pred_test))
    df_train_errors = pd.DataFrame(d_train_errors)
    df_train_errors.set_index('Number of components to keep', inplace=True)
    df_test_errors = pd.DataFrame(d_test_errors)
    df_test_errors.set_index('Number of components to keep', inplace=True)
    df_train_times = pd.DataFrame(d_train_times)
    df_train_times.set_index('Number of components to keep', inplace=True)
    df_test_times = pd.DataFrame(d_test_times)
    df_test_times.set_index('Number of components to keep', inplace=True)
    styles = ['p-', 's-', 'o-', '^-','k--']
    filename = dataset+"_exp5_"+c_alg_name
    plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    df_train_errors.plot(ax=axes[0,0], style=styles)
    axes[0,0].set_ylabel("Training Accuracy")
    df_test_errors.plot(ax=axes[0,1], style=styles)
    axes[0,1].set_ylabel("Testing Accuracy")
    df_train_times.plot(ax=axes[1,0], style=styles)
    axes[1,0].set_ylabel("Training Time")
    df_test_times.plot(ax=axes[1,1], style=styles)
    axes[1,1].set_ylabel("Testing Time")
    axes[0,0].legend_.set_visible(False)
    axes[0,1].legend_.set_visible(False)
    axes[1,1].legend_.set_visible(False)
    axes[1,0].legend_.set_visible(False)
    handles, labels = axes[0,0].get_legend_handles_labels()
    plt.figlegend(handles=handles, labels=labels, loc='upper right', prop=fontP)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.suptitle('NN on '+c_alg_name+'-Clustered Reduced Data for '+dataset)
    plt.savefig(filename)

if __name__=='__main__':
    np.random.seed(42)
    dataset1 = 'car.data' # 6 attributes 4 classes
    print("-----------------------------------Dataset 1--------------------------------------")
    x, y = load_data(dataset1)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.3)
    dataset1 = dataset1.replace('.', '_')
#    print(y)
    
    print("--------------------------Experiment 1------------------------------")
    ks = [2,3,4,5,6,7]
    clustering(dataset1, x, y, ks)

    print("--------------------------Experiment 2------------------------------")
    ns = [1,2,3,4,5,6]
    dimension_reduction(dataset1, x, y, ns)

    print("--------------------------Experiment 3------------------------------")
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(x)
    exp3(dataset1, 4, X, y, ns)
    plot_pro_1(X, y, LinearDiscriminantAnalysis ,'LDA')
    plot_pro_1(X, y, PCA ,'PCA')

    print("-----------------------------------Experiment 4--------------------------------------")
    exp4(dataset1, x_train, x_test, y_train, y_test, ns) 

    print("-----------------------------------Experiment 5--------------------------------------")
    exp5(dataset1, x_train, x_test, y_train, y_test, KMeans, 'Kmeans', n_classes=4, ns=ns)
    exp5(dataset1, x_train, x_test, y_train, y_test, GaussianMixture, 'EM', n_classes=4, ns=ns)

    dataset2 = 'tic-tac-toe.data' # 9 attributes 2 classes
    print("-----------------------------------Dataset 2--------------------------------------")
    x, y = load_data(dataset2,attributes=10)
    x_train, x_test, y_train, y_test = split_train_test(x, y, 0.2)
    dataset2 = dataset2.replace('.', '_')
#    print(y)

    print("--------------------------Experiment 1------------------------------")
    ks = [2,3,4,5]
    clustering(dataset2, x, y, ks)

    print("--------------------------Experiment 2------------------------------")
    ns = [1,2,3,4,5,6,7,8,9]
    dimension_reduction(dataset2, x, y, ns)

    print("--------------------------Experiment 3------------------------------")
    X = scaler.fit_transform(x)
    exp3(dataset2, 2, X, y, ns)
    plot_pro_2(X, y, PCA, 'PCA')
    plot_pro_2(X, y, GaussianRandomProjection, 'Randomized Projection')
