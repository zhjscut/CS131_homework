import numpy as np
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from numpy import linalg
from utils import pad, unpad
import copy
import random

    
def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))
    response = np.zeros((H, W))
    
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)
    dx_2 = dx ** 2
    dy_2 = dy ** 2
    dx_dy = dx * dy
    pad_size = window_size // 2
    pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    dx_2, dy_2, dx_dy = np.pad(dx_2, pad_width, mode='edge'), np.pad(dy_2, pad_width, mode='edge'), np.pad(dx_dy, pad_width, mode='edge')
    for i in range(0, H):
        for j in range(0, W):
            Dx_2 = np.sum(window * dx_2[i: i+window_size, j: j+window_size])
            Dy_2 = np.sum(window * dy_2[i: i+window_size, j: j+window_size])
            Dx_Dy = np.sum(window * dx_dy[i: i+window_size, j: j+window_size])
            M = np.array([[Dx_2, Dx_Dy], [Dx_Dy, Dy_2]])                    
            response[i, j] = np.linalg.det(M) - k * (np.trace(M) ** 2)
    return response

def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = patch.flatten()
    feature = feature - feature.mean()
    if feature.std() < 1e-5:
        pass #方差为0就不动它了
    else:
        feature /= feature.std()

    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    print(np.array(desc).shape)
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    #当最近距离与第二近距离之比小于threshold时，认为是“显著近”，也即找到了匹配点
    matches = []
    
    N = desc1.shape[0]
    dists = cdist(desc1, desc2) #输出dists的第n行保存了desc1第n个向量与desc2所有向量的距离
    for i in range(0, N):        
        dist = dists[i, :]
        index = np.argsort(dist)
        if dist[index[0]] / dist[index[1]] < threshold:
            match = np.array([i, index[0]])
            if isinstance(matches, list):
                matches = copy.deepcopy(match)
            else:
                matches = np.vstack((matches, match))
    
    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)
    result = linalg.lstsq(p2, p1) #求解从p1变换到p2的最优变换矩阵
    H = result[0]
    #变换矩阵X是一个3×3的矩阵
    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)                
    matched1 = keypoints1[matches[:,0]]
    matched2 = keypoints2[matches[:,1]]
    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    for iter in range(0, n_iters): 
        selected_samples = random.sample(range(N), n_samples)
        match1 = matched1[selected_samples, :]
        match2 = matched2[selected_samples, :]
        H = fit_affine_matrix(match1, match2) #让match2的点经变换变到match1
        result = pad(matched2).dot(H)[:,0:2] #变换结果依然是N×3矩阵
        dists = np.sqrt(np.sum((result - matched1) ** 2, axis=1))
        if np.sum(dists <= threshold) > n_inliers: #有更好的拟合结果，更新一下
            n_inliers = np.sum(dists <= threshold)
            print(n_inliers)
            max_inliers = np.where(dists <= threshold)
    match1 = matched1[max_inliers]
    match2 = matched2[max_inliers]
    H = fit_affine_matrix(match1, match2)
    result = pad(matched2).dot(H)[:,0:2]
    dists = np.sqrt(np.sum((result - matched1) ** 2, axis=1))
    n_inliers = np.sum(dists <= threshold)
    print(n_inliers)
    max_inliers = np.where(dists <= threshold)
    
    return H, matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. compute the gradient image in x and y (already done for you)
    2. compute gradient histograms
    3. normalize across block 
    4. flattening block into a feature vector

    Args:
        patch: grayscale image patch of shape (h, w)
        pixels_per_cell: size of a cell with shape (m, n)

    Returns:
        block: 1D array of shape ((h*w*n_bins)/(m*n))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)
   
    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180
    
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell) #返回一个4维矩阵，表示一共有m行n列个cell,每个cell的维度是x行y列
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]
    cells = np.zeros((rows, cols, n_bins))
    # Compute histogram per cell
    for i in range(0, rows):
        for j in range(0, cols):
            for n_bin in range(0, n_bins):
                cells[i, j, n_bin] = np.sum(G_cells[i, j] * ((theta_cells[i, j] >= n_bin * degrees_per_bin) & (theta_cells[i, j] < (n_bin+1) * degrees_per_bin)))
    block = cells.flatten()
    block = block - block.mean()
    if block.std() < 1e-5:
        pass #方差为0就不动它了
    else:
        block /= block.std()

    return block