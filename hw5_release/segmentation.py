import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)             
    dist_center = np.zeros((k, N)) #我自己加上去的
    # print('init: ', centers)
    from scipy.spatial.distance import cdist #违规操作，没办法了pdist内存不够用
    for n in range(num_iters):
        ### YOUR CODE HERE
        
        center_old = centers.copy()
        # distance = pdist(np.vstack((centers, features))) #N个数据点进行pdist返回的距离向量的长度是(N+k)(N+k-1)/2 [0:k, 4:] #我们只需要X与质心的距离矩阵，大小为k*N
        # print(distance)
        # count = 0
        # for i in range(0, k): #丢掉质心之间的k(k-1)/2个距离值和点之间的N(N-1)/2个点，取出质心与点之间的距离向量
            # count += k-i-1
            # dist_center[i, :] = distance[count:count+N]
            # count += N
        # # print(dist_center)    
        # dist_center = squareform(pdist(np.vstack((centers, features))))[0:k, k:]
        dist_center = cdist(centers, features)
        assignments = np.argmin(dist_center, axis=0)
        a = np.zeros(k)
        for i in range(0, k):
            # print(features.shape)
            # print(np.mean(features[assignments==i], axis=0))
            centers[i, :] = np.mean(features[assignments==i], axis=0)
        if np.sum((center_old - centers) ** 2) < 1e-5:
            print('total turns: ',n)
            break
        # print('turn ', n, ': ',centers)

        ### END YOUR CODE
        
    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        pass
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to defeine distance between two clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N
    
    distance = squareform(pdist(features)) #我自己加上去的
    #似乎要先把自己对自己的距离设为无穷大，不然会找到自己为最近点的
    distance[np.arange(N), np.arange(N)] = 1e5
    count1, count2 = 0, 0
    while n_clusters > k:
        ### YOUR CODE HERE
        
        print('new turn:',n_clusters)
        for i in range(0, n_clusters):
            if i != assignments[i]: #每回合只有这个类的老大才能出去找一处人，小弟都不能找人，不然大种群会
                print('continue:',n_clusters) 
                continue
            print(np.unique(assignments).shape)
            print('i = ',i, 'n_clusters = ',n_clusters)
            # distance[i, assignments==i] = 1e5 #先把自己对与自己同类点的距离设为无穷大，不然找到的最近点有可能是自己人
            # if i==0:
                # print(distance[i, :])
                
            index = assignments[np.argmin(distance[i, :])] #找到离自己最近的点所属的类
            print('I am ',i, 'I find ',index)
            if i < index: #约定归为序号小的类号
                # print(distance[:, assignments==index])
                # distance[:, assignments==index] = 1e5 #可以考虑把被聚类后失去自己标签的点对其他质心的距离设为无穷大，以消除该列对其他点计算最近距离时的影响                
                # print(distance[:, assignments==index])
                assignments[assignments==index] = i #合并两类点的assignments
                # print(assignments[assignments==i])
                # print(distance[:, assignments==i])
                centers[i, :] = np.mean(features[assignments==i], axis=0) #计算新的质心并更新centers（失去自己标签的点的centers是没有更新的，节省时间）
                distance[:, i] = np.sum(np.sqrt((features - centers[i, :]) ** 2), axis=1) #之所以只需要修改一列，是因为同属一个index的其他点到各点的距离已经在之前的合并过程中被置为无穷大了
                distance[assignments==i, i] = distance[assignments==i, i] + 1e5 #把自己对与自己同类点的距离设为无穷大，不然找到的最近点有可能是自己人
                distance[i, assignments==i] = distance[i, assignments==i] + 1e5
                n_clusters -= 1
                count1 += 1
            else:
                distance[:, assignments==i] = 1e5 #可以考虑把被聚类后失去自己标签的点对其他质心的距离设为无穷大，以消除该列对其他点计算最近距离时的影响
                assignments[assignments==i] = index #合并两类点的assignments
                centers[index, :] = np.mean(features[assignments==index], axis=0) #计算新的质心并更新centers（失去自己标签的点的centers是没有更新的，节省时间）
                distance[:, index] = np.sum(np.sqrt((features - centers[index, :]) ** 2), axis=1) #之所以只需要修改一列，是因为同属一个index的其他点到各点的距离已经在之前的合并过程中被置为无穷大了
                distance[assignments==index, index] = distance[assignments==index, index] + 1e5 #把自己对与自己同类点的距离设为无穷大，不然找到的最近点有可能是自己人
                distance[index, assignments==index] = distance[index, assignments==index] + 1e5
                n_clusters -= 1 
                count2 += 1
            if n_clusters <= k:
                break
        print(assignments)
        counts = np.zeros(N)
        for j in range(0, N):
            counts[j] = np.sum(assignments==j)
        print(counts)
        ### END YOUR CODE
    assignments_tmp = assignments.copy() #后面这部分也是我加上去的
    index_list = np.unique(assignments)
    print(index_list)
    print(count1, count2)
    for i in range(0, k):
        assignments[assignments_tmp==index_list[i]] = i

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    for i in range(H):
        features[i*W : (i+1)*W, :] = img[i, :, :]
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    for i in range(H):
        features[i*W : (i+1)*W, 0:3] = img[i, :, :]
        features[i*W : (i+1)*W, 3:] = np.hstack((np.ones((W, 1)) * i, np.arange(W).reshape(W, 1)))
    for i in range(5):
        features[:, i] = (features[:, i] - features[:, i].mean()) / features[:, i].std()    
    ### END YOUR CODE
    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features
    

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    ### YOUR CODE HERE
    size = mask_gt.shape[0] * mask_gt.shape[1]
    diff = mask_gt - mask
    accuracy = np.sum(diff.flatten()==0) / size
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy