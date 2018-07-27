import numpy as np
import copy

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    row, col = kernel.shape
    row_add, col_add = int(row / 2), int(col / 2)
    zeros_row = np.zeros((row_add, image.shape[1]))
    image0 = np.vstack((zeros_row, image, zeros_row))#np.hstack实现二维矩阵的行合并（列扩展）
    zeros_col = np.zeros((image0.shape[0], col_add))
    image0 = np.hstack((zeros_col, image0, zeros_col))#np.vstack实现二维矩阵的列合并（行扩展）
    result = np.zeros(image.shape)
    kernel0 = copy.deepcopy(kernel)
    kernel1 = np.zeros(kernel.shape)
    for i in range(0, kernel.shape[0]):        
        kernel0[i, :] = kernel[kernel.shape[0] - 1 - i, :]
    for j in range(0, kernel.shape[1]):
        kernel1[:, j] = kernel0[:, kernel.shape[1] - 1 - j]
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            X = image0[i: i + row, j: j + col]
            result[i, j] = np.sum(X * kernel1)
    return result

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """
    H, W = image.shape
    zeros_row = np.zeros((pad_height, image.shape[1]))
    result = np.vstack((zeros_row, image, zeros_row))#np.hstack实现二维矩阵的行合并（列扩展）
    zeros_col = np.zeros((result.shape[0], pad_width))
    result = np.hstack((zeros_col, result, zeros_col))#np.vstack实现二维矩阵的列合并（行扩展）    
    return result
    
    

def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """   
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    pad_height, pad_width = int(Hk / 2), int(Wk / 2)
    image0 = zero_pad(image, pad_height, pad_width)
    kernel0 = copy.deepcopy(kernel)
    kernel1 = np.zeros(kernel.shape)
    result = np.zeros((Hi, Wi))
    for i in range(0, Hk):        
        kernel0[i, :] = kernel[kernel.shape[0] - 1 - i, :]
    for j in range(0, Wk):
        kernel1[:, j] = kernel0[:, kernel.shape[1] - 1 - j]
    for i in range(0, Hi):
        for j in range(0, Wi):
            X = image0[i: i + Hk, j: j + Wk]
            result[i, j] = np.sum(X * kernel1)
    return result    
    
    

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(image, kernel):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    pad_height, pad_width = int(Hk / 2), int(Wk / 2)
    image0 = zero_pad(image, pad_height, pad_width)
    result = np.zeros((Hi, Wi))
    for i in range(0, Hi):
        for j in range(0, Wi):
            X = image0[i: i + Hk, j: j + Wk]
            result[i, j] = np.sum(X * kernel)
    return result  
    

def zero_mean_cross_correlation(image, kernel):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    pad_height, pad_width = int(Hk / 2), int(Wk / 2)
    image0 = zero_pad(image, pad_height, pad_width)
    result = np.zeros((Hi, Wi))
    kernel1 = copy.deepcopy(kernel)
    kernel1 = kernel1 - kernel1.mean()
    for i in range(0, Hi):
        for j in range(0, Wi):
            X = image0[i: i + Hk, j: j + Wk]
            result[i, j] = np.sum(X * kernel1)
    return result  

def normalized_cross_correlation(image, kernel):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    pad_height, pad_width = int(Hk / 2), int(Wk / 2)
    image0 = zero_pad(image, pad_height, pad_width)
    result = np.zeros((Hi, Wi))
    kernel1 = copy.deepcopy(kernel)
    kernel1 = (kernel1 - kernel1.mean()) / kernel1.var()
    for i in range(0, Hi):
        for j in range(0, Wi):
            X = image0[i: i + Hk, j: j + Wk]
            X = (X - X.mean()) / X.var()            
            result[i, j] = np.sum(X * kernel1)
    return result 
