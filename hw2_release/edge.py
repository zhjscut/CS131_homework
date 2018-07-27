import numpy as np
import math
import copy

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2 
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')        
    kernel0 = copy.deepcopy(kernel)
    kernel1 = np.zeros(kernel.shape)
    result = np.zeros((Hi, Wi))
    for i in range(0, Hk):        
        kernel0[i, :] = kernel[kernel.shape[0] - 1 - i, :]
    for j in range(0, Wk):
        kernel1[:, j] = kernel0[:, kernel.shape[1] - 1 - j]
    for i in range(0, Hi):
        for j in range(0, Wi):
            X = padded[i: i + Hk, j: j + Wk]
            result[i, j] = np.sum(X * kernel1)
    return result 

   
def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))
    k = math.floor(size / 2)  
    for i in range(0, size):
        for j in range(0, size):
            kernel[i, j] = 1 / 2 / math.pi / (sigma ** 2) * math.exp(-((i-k)**2 + (j-k)**2) / 2 / sigma**2)

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """
    kernel = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]) #公式可以直接拆成相关的形式，但是把变换核翻转一下，从相关变到卷积
    out = conv(img, kernel)
    
    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """
    kernel = np.array([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]])
    out = conv(img, kernel)

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy arr   ay of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    G_x = partial_x(img)
    #G_x[G_x < 1e-10] = 1e-10
    G_y = partial_y(img)
    G = np.sqrt(G_x ** 2 + G_y ** 2)
    theta = np.arctan2(G_y, G_x) #得到的是弧度值，范围-π~π
    theta[theta < 0] = theta[theta < 0] + 2 * math.pi
    theta = theta / np.pi * 180
    
    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) #四舍五入
    G_tmp = np.pad(G,((1, 1),(1, 1)), 'constant')
    # 在横向为y纵向为x的坐标系中，矩阵的行号就是x坐标，列号就是y坐标，如果不是就要倒过来（下面的式子中是横x纵y，如果遇到横y纵x的情况，就把0跟90对应的情况要互换，135不用改，45要改）
    for i in range(1, H+1):
        for j in range(1, W+1):
            if (theta[i - 1, j - 1] == 0 or theta[i - 1, j - 1] == 4 or theta[i - 1, j - 1] == 8): 
                if (G_tmp[i, j] > G_tmp[i, j + 1] and G_tmp[i, j] > G_tmp[i, j - 1]):
                    out[i - 1, j - 1] = G_tmp[i, j] #减1是因为边缘零填充之后坐标产生了变化
            elif (theta[i - 1, j - 1] == 1 or theta[i - 1, j - 1] == 5):
                if (G_tmp[i, j] > G_tmp[i + 1, j + 1] and G_tmp[i, j] > G_tmp[i - 1, j - 1]):
                    out[i - 1, j - 1] = G_tmp[i, j]                            
            elif (theta[i - 1, j - 1] == 2 or theta[i - 1, j - 1] == 6):
                if (G_tmp[i, j] > G_tmp[i + 1, j] and G_tmp[i, j] > G_tmp[i - 1 , j]):
                    out[i - 1, j - 1] = G_tmp[i, j]               
            #elif (theta[i - 1, j - 1] == 3 or theta[i - 1, j - 1] == 7):
            else:
                if (G_tmp[i, j] > G_tmp[i - 1, j + 1] and G_tmp[i, j] > G_tmp[i + 1, j - 1]):
                    out[i - 1, j - 1] = G_tmp[i, j]                
                
    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower thresholdc
    """
    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)
    strong_edges = (img >= high) #原：> < > 1.>= < > 2.>= < >= 3. > <= > 4.> <= >=
    weak_edges = (img < high) * (img >= low)

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T #indices是个二维数组，每一行保存一个强边缘点的坐标
    edges = copy.deepcopy(strong_edges)
    for n in range(0, indices.shape[0]):
        neighbors = get_neighbors(indices[n, 0], indices[n, 1], H, W)
        for location in neighbors:
            if weak_edges[location[0], location[1]] == 1:
                edges[location[0], location[1]] = 1

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)

    return edge
    #return kernel, smoothed, G, theta, nms, strong_edges, weak_edges, edge



def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    # for i in range(0, len(xs)):
        # if i % 1000 == 0:
            # print(i)
        # rho = np.round(xs[i] * cos_t + ys[i] * sin_t).astype(int) + diag_len
        # for j in range(0, num_thetas):
            # accumulator[rho[j], j] += 1

    for i in range (0, num_thetas):
        if i % 45 == 0:
            print(i)
        rho = np.round(xs * cos_t[i] + ys * sin_t[i]).astype(int) + diag_len
        for j in range(0, len(rho)):
            accumulator[rho[j], i] += 1                   
            
    return accumulator, rhos, thetas
