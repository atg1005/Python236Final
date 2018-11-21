import numpy as np

def get_cluster_data_points(cluster_labels, orig_data, cluster_num):
    """
    Returns np array of data points given the dataset, cluster labels and a cluster number.
    """
    return orig_data[np.where(cluster_labels == cluster_num)[0]]

def max_of_array(np_data):
    """
    Returns the maximum Y  value found within a given list of xy data values.
    """
    maxP = np_data[0][1]
    for point in np_data:
        #print(point[1])
        if maxP < point[1]:
            maxP = point[1]
    return maxP

def max_center_index(cluster_centers):
    """
    Given cluster center points return the index of the maximum.
    This is the cluster number we want to search to find overall maximum.
    Very similar to max_of_array but we dont want to always maintain index
    when seraching for max which makes this function necessary.
    """
    index = 0
    max_index = 0
    maxP = cluster_centers[0][1]
    for point in cluster_centers:
        if maxP < point[1]:
            maxP = point[1]
            max_index = index
        index += 1
    return max_index

def find_max_with_centers(labels,centers,data):
    """
    Given clustering labels, centers, and data return the maximum value found.
    """
    return max_of_array(get_cluster_data_points(labels,data,max_center_index(centers)))

if __name__ == '__main__':
    print('Module contains functions for finding the maximum.')
