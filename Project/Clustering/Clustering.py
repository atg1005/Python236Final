import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.cluster import AffinityPropagation, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def readData(pathToData,title1,title2):
    """
    given path to data create a pandas data frame and populate
    a list of lists which is returned.
    """
    data = []
    dataFrame = pd.read_excel(open(pathToData,'rb'))
    for index,row in dataFrame.iterrows():
        temp = []
        temp.append(row[title1])
        temp.append(row[title2])
        data.append(temp)
    return data

def KMeans_Cluster(np_data, num_clusters=3):
    """
    Perfrom k-means clustering returns cluster labels.
    """
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(np_data)
    return kmeans.predict(np_data)

def Hierarchical_Cluster(np_data, num_clusters=3):
    """
    Performs Hierarchical clustering returns only labels.
    """
    h_cluster =  AgglomerativeClustering(n_clusters=num_clusters, affinity = 'euclidean', linkage = 'ward')
    return h_cluster.fit_predict(np_data)


def Gaussian_Mixture_Model_Clustering(np_data, num_clusters=3):
    """
    perfroms GMM clustering returns labels.
    """
    gmm = GaussianMixture(n_components=num_clusters, init_params='kmeans')
    gmm.fit(np_data)
    return gmm.predict(np_data)

def DBSCAN_Cluster(np_data,max_dist=0.01,min_neighbors=4):
    """
    Performs DBSCAN clustering returns labels and dbscan object.
    """
    dbscan = DBSCAN(eps=max_dist, min_samples=min_neighbors)
    return dbscan.fit_predict(np_data)

def Mean_Shift_Cluster(np_data):
    """
    Performs Mean Shift clustering returns labels.
    """
    mean_shift = MeanShift()
    return mean_shift.fit_predict(np_data)

def Affinity_Propagation_Cluster(np_data):
    """
    Performs affinity propagation clustering and returns labels.
    """
    ap = AffinityPropagation()
    return ap.fit_predict(np_data)

def Spectrial_Cluster(np_data,num_clusters=3):
    """
    Performs Spectrial clustering and returns labels.
    **Gives warning via console may not work as expected.
    """
    spectrial = SpectralClustering(n_clusters=num_clusters)
    return spectrial.fit_predict(np_data)

def Birch_Cluster(np_data,num_clusters=3):
    """
    Perform birch clustering and return labels
    """
    birch = Birch()
    return birch.fit_predict(np_data)


if __name__ == '__main__':
    centers = [[5,5],[-5,5],[5,-5],[-5,5]] # for make_blobs
    num_clusters = 4
    # read in data and convert to an np array
    #data_points = np.asarray(readData('../Data/randomPoints.xlsx','X','Y'),dtype=np.int32)
    #data_points = datasets.make_circles()[0]
    data_points = datasets.make_moons(50)[0]
    #data_points = datasets.make_blobs(100,centers=centers)[0]

    # Perform various clustering methods
    kmeans_labels = KMeans_Cluster(data_points,num_clusters)
    hierarchical_labels = Hierarchical_Cluster(data_points,num_clusters)
    gmm_labels = Gaussian_Mixture_Model_Clustering(data_points, num_clusters)
    dbscan_labels = DBSCAN_Cluster(data_points,max_dist=0.001)
    mean_shift_labels = Mean_Shift_Cluster(data_points)
    ap_labels = Affinity_Propagation_Cluster(data_points)
    spectrial_labels = Spectrial_Cluster(data_points, num_clusters)
    birch_labels = Birch_Cluster(data_points,num_clusters)

    # Graph clustering
    fig  = plt.figure(figsize = (12,7)) # found size by trial and error
    fig.subplots_adjust(hspace = 0.85)
    fig.subplots_adjust(wspace = 0.39)
    fig.subplots_adjust(left = 0.08)
    fig.subplots_adjust(right = 0.97)
    fig.suptitle('Clustering')

    p1 = fig.add_subplot(241)
    p2 = fig.add_subplot(242)
    p3 = fig.add_subplot(243)
    p4 = fig.add_subplot(244)
    p5 = fig.add_subplot(245)
    p6 = fig.add_subplot(246)
    p7 = fig.add_subplot(247)
    p8 = fig.add_subplot(248)

    p1.scatter(data_points[:,0] , data_points[:,1] , c = kmeans_labels, s=50, cmap='viridis')
    p1.set_title('K-Means (' + str(num_clusters) + ')')
    p1.set_xlabel('X')
    p1.set_ylabel('Y',rotation=0)

    p2.scatter(data_points[:,0] , data_points[:,1] , c = hierarchical_labels, s=50, cmap='viridis')
    p2.set_title('Hierarchical (' + str(num_clusters) + ')')
    p2.set_xlabel('X')
    p2.set_ylabel('Y',rotation=0)

    p3.scatter(data_points[:,0] , data_points[:,1] , c = gmm_labels, s=50, cmap='viridis')
    p3.set_title('Guassian Mixture Model (' + str(num_clusters) + ')')
    p3.set_xlabel('X')
    p3.set_ylabel('Y',rotation=0)

    # DBSCAN is not as easy to plot and I could not plot it correctly.
    # below is dbscan visualization from sklearn I did not write this.
    # for moon data use 0.3
    db = DBSCAN(eps=0.3, min_samples=4).fit(data_points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = data_points[class_member_mask & core_samples_mask]
        p4.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)
        xy = data_points[class_member_mask & ~core_samples_mask]
        p4.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)
    #p4.scatter(data_points[:,0] , data_points[:,1] , c = dbscan_labels, s=50, cmap='viridis')
    p4.set_title('DBSCAN')
    p4.set_xlabel('X')
    p4.set_ylabel('Y',rotation=0)

    p5.scatter(data_points[:,0] , data_points[:,1] , c = mean_shift_labels, s=50, cmap='viridis')
    p5.set_title('Mean Shift')
    p5.set_xlabel('X')
    p5.set_ylabel('Y',rotation=0)

    p6.scatter(data_points[:,0] , data_points[:,1] , c = ap_labels, s=50, cmap='viridis')
    p6.set_title('Affinity Propagation')
    p6.set_xlabel('X')
    p6.set_ylabel('Y',rotation=0)

    p7.scatter(data_points[:,0] , data_points[:,1] , c = spectrial_labels, s=50, cmap='viridis')
    p7.set_title('Spectrial (' + str(num_clusters) + ')')
    p7.set_xlabel('X')
    p7.set_ylabel('Y',rotation=0)

    p8.scatter(data_points[:,0] , data_points[:,1] , c = birch_labels, s=50, cmap='viridis')
    p8.set_title('Birch')
    p8.set_xlabel('X')
    p8.set_ylabel('Y',rotation=0)

    plt.show()
