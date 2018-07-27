import numpy as np
import copy

def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    A = image.copy()
    eigenvalues1, U = np.linalg.eig(np.dot(A, A.T))
    eigenvalues1[eigenvalues1<0] = 0
    sigma = np.sqrt(eigenvalues1)
    index = np.flipud(np.argsort(sigma))
    sigma = sigma[index]
    U = U[:, index]
    _, V = np.linalg.eig(np.dot(A.T, A)) #提取主分量的时候V是不能动的
    # U = U[:, 0:num_values]
    # sigma = sigma[0:num_values]   
    # sigma = np.diag(sigma)
    # print('函数开始')
    # print(U)
    # print(sigma)
    # print(V)
    # if sigma.shape[1] < V.shape[0]:
        # sigma = np.hstack((sigma, np.zeros((sigma.shape[0], V.shape[0] - sigma.shape[1]))))
        # # 如果sigma大，两种都实验一下，一种是裁剪sigma，一种是补V
    # elif sigma.shape[1] > V.shape[0]:
        # # V = np.hstack((V, np.zeros((V.shape[0], sigma.shape[1] - V.shape[0]))))
        # sigma = sigma[:, 0: V.shape[0]-sigma.shape[1]]
    # if sigma.shape[0] < U.shape[1]:
        # sigma = np.vstack((sigma, np.zeros((u.shape[1] - sigma.shape[0], sigma.shape[1]))))   
    # print(U)
    # print(sigma)
    # print(V)
    # print(U.shape, sigma.shape, V.shape) 
    # # print(U, sigma, V)
    # print(U.shape, sigma.shape, V.shape)
    # print(np.dot(np.dot(U, sigma), V.T))
    # print('函数结束')
    U, sigma, V = np.linalg.svd(image) #先作弊一下下


    U = U[:, 0:num_values]
    sigma = sigma[0:num_values]  
    V = V[0:num_values, :]
    sigma1 = sigma.copy()
    sigma = np.diag(sigma)
    if sigma.shape[1] < V.shape[0]: #最后要把这两段放进我的svd函数代码里。svd的重建有时是需要补全零行或全零列的！
        sigma = np.hstack((sigma, np.zeros((sigma.shape[0], V.shape[0] - sigma.shape[1]))))
    if sigma.shape[0] < U.shape[1]:
        sigma = np.vstack((sigma, np.zeros((U.shape[1] - sigma.shape[0], sigma.shape[1]))))        
    compressed_image = np.dot(np.dot(U, sigma), V)
    compressed_size = U.size + V.size + sigma1.size
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
