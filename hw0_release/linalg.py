import numpy as np


def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (x, n)
        vector2: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x,x) (scalar if x = 1)
    """
    out = None
    if len(vector1.shape) == 1 and len(vector2.shape) == 1:
        if vector1.shape[0] == 1 or vector2.shape[0] ==1:
            out = np.array(vector1 * vector2)
            return out
        elif vector1.shape[0] != vector2.shape[0]:
            raise ValueError('invalid value!')
            return None
        out = np.array([np.sum(vector1 * vector2)])
    elif len(vector1.shape) == 1: #if v1 is a vector
        out = np.zeros(vector2.shape[1])
        for i in range(vector2.shape[1]):
            out[i] = np.sum(vector1 * vector2[:, i])
        return out
    elif len(vector2.shape) == 1: #if v2 is a vector
        # print(vector1.shape)
        out = np.zeros(vector1.shape[0])
        for i in range(vector1.shape[0]):
            out[i] = np.sum(vector1[i, :] * vector2)
        return out
    elif vector1.shape[1] != vector2.shape[0]: #both of them are matrix
        raise ValueError('invalid value!')
    else:
        out = np.zeros([vector1.shape[0], vector2.shape[1]])
        for i in range(vector1.shape[0]):
            for j in range(vector2.shape[1]):
                out[i, j] = np.sum(vector1[i, :] * vector2[:, j])
    return out

def matrix_mult(M, vector1, vector2):
    """ Implement (vector1.T * vector2) * (M * vector1)
    Args:
        M: numpy matrix of shape (x, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (1, x)
    """
    out = None
    out1 = dot_product(vector1, vector2)
    out2 = dot_product(M, vector1)
    print(out1, out2)
    out = dot_product(out1, out2)

    return out

def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    u = None
    s = None
    v = None
    v0 = dot_product(matrix.T, matrix)
    eigen = np.linalg.eig(v0)
    eigen_vals = eigen[0]
    v = eigen[1] #v is the eigen_vector
    u = dot_product(matrix, v)
    new_orth_len = np.zeros([u.shape[1], u.shape[1]])
    orth_sq = dot_product(u.T, u)

    for i in range(u.shape[1]):
        for j in range(u.shape[0]):
            u[i][j] /= (orth_sq[j][j] ** 0.5)
        new_orth_len[j][j] = orth_sq[j][j] ** 0.5


    return u, s, v

def gen_inv(a):
    a_sq = np.dot(a.T, a)
    eigen = np.linalg.eig(a_sq)
    eigen_vals = eigen[0]
    eigen_vectors = eigen[1]

    orth = a.dot(eigen_vectors)
    new_orth_len = np.zeros([orth.shape[1], orth.shape[1]])
    orth_sq = orth.T.dot(orth)

    for j in range(orth.shape[1]):
        for i in range(orth.shape[0]):
            orth[i][j] /= (orth_sq[j][j] ** 0.5)
        new_orth_len[j][j] = orth_sq[j][j] ** 0.5

    return (orth, new_orth_len, eigen_vectors)

def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output
        
    Returns:
        singular_values: array of shape (n)
    """
    singular_values = None
    u, s, v = svd(matrix)
    values = np.diag(s)
    values = sorted(values, reverse = True)
    singular_values = values[0:n]
    return singular_values

def eigen_decomp(matrix):
    """ Implement Eigen Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, )

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    """
    w = None
    v = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return w, v

def get_eigen_values_and_vectors(matrix, num_values):
    """ Return top n eigen values and corresponding vectors of matrix
    Args:
        matrix: numpy matrix of shape (m, m)
        num_values: number of eigen values and respective vectors to return
        
    Returns:
        eigen_values: array of shape (n)
        eigen_vectors: array of shape (m, n)
    """
    w, v = eigen_decomp(matrix)
    eigen_values = []
    eigen_vectors = []
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return eigen_values, eigen_vectors
