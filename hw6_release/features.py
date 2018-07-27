import numpy as np
import scipy
import scipy.linalg


class PCA(object):
    """Class implementing Principal component analysis (PCA).

    Steps to perform PCA on a matrix of features X:
        1. Fit the training data using method `fit` (either with eigen decomposition of SVD)
        2. Project X into a lower dimensional space using method `transform`
        3. Optionally reconstruct the original X (as best as possible) using method `reconstruct`
    """

    def __init__(self):
        self.W_pca = None
        self.mean = None

    def fit(self, X, method='svd'):
        """Fit the training data X using the chosen method.

        Will store the projection matrix in self.W_pca and the mean of the data in self.mean

        Args:
            X: numpy array of shape (N, D). Each of the N rows represent a data point.
               Each data point contains D features.
            method: Method to solve PCA. Must be one of 'svd' or 'eigen'.
        """
        _, D = X.shape
        self.mean = None   # empirical mean, has shape (D,)
        X_centered = None  # zero-centered data

        # YOUR CODE HERE
        # 1. Compute the mean and store it in self.mean
        # 2. Apply either method to `X_centered`
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        if method == 'svd':
            self.W_pca, vals = self._svd(X_centered)
        else:
            self.W_pca, vals = _eigen_decomp(X_centered)
        # END YOUR CODE

        # Make sure that X_centered has mean zero
        assert np.allclose(X_centered.mean(), 0.0)

        # Make sure that self.mean is set and has the right shape
        assert self.mean is not None and self.mean.shape == (D,)

        # Make sure that self.W_pca is set and has the right shape
        assert self.W_pca is not None and self.W_pca.shape == (D, D)

        # Each column of `self.W_pca` should have norm 1 (each one is an eigenvector)
        for i in range(D):
            assert np.allclose(np.linalg.norm(self.W_pca[:, i]), 1.0)

    def _eigen_decomp(self, X):
        """Performs eigendecompostion of feature covariance matrix.

        Args:
            X: Zero-centered data array, each ROW containing a data point.
               Numpy array of shape (N, D).

        Returns:
            e_vecs: Eigenvectors of covariance matrix of X. Eigenvectors are
                    sorted in descending order of corresponding eigenvalues. Each
                    column contains an eigenvector. Numpy array of shape (D, D).
            e_vals: Eigenvalues of covariance matrix of X. Eigenvalues are
                    sorted in descending order. Numpy array of shape (D,).
        """
        N, D = X.shape
        e_vecs = None
        e_vals = None
        # YOUR CODE HERE
        # Steps:
        #     1. compute the covariance matrix of X, of shape (D, D)
        #     2. compute the eigenvalues and eigenvectors of the covariance matrix
        #     3. Sort both of them in decreasing order (ex: 1.0 > 0.5 > 0.0 > -0.2 > -1.2)
        covar = np.dot(X.T, X) / (N-1)
        e_vals, e_vecs = np.linalg.eig(covar)
        index = np.flipud(np.argsort(e_vals))
        e_vals = e_vals[index]
        e_vecs = e_vecs[:,index]
        # END YOUR CODE

        # Check the output shapes
        assert e_vals.shape == (D,)
        assert e_vecs.shape == (D, D)

        return e_vecs, e_vals

    def _svd(self, X):
        """Performs Singular Value Decomposition (SVD) of X.

        Args:
            X: Zero-centered data array, each ROW containing a data point.
                Numpy array of shape (N, D).
        Returns:
            vecs: right singular vectors. Numpy array of shape (D, D)
            vals: singular values. Numpy array of shape (K,) where K = min(N, D)
        """
        vecs = None  # shape (D, D)
        N, D = X.shape
        vals = None  # shape (K,)
        # YOUR CODE HERE
        # Here, compute the SVD of X
        # Make sure to return vecs as the matrix of vectors where each column is a singular vector
        _, vals, vecs_T = scipy.linalg.svd(X)
        vecs = vecs_T.T
        # num_values = U.shape[1]
        # U = U[:, 0:num_values]
        # sigma = sigma[0:num_values]  
        # V = V[0:num_values, :]

        # sigma = np.diag(sigma)
        # if sigma.shape[1] < V.shape[0]: #最后要把这两段放进我的svd函数代码里。svd的重建有时是需要补全零行或全零列的！
            # sigma = np.hstack((sigma, np.zeros((sigma.shape[0], V.shape[0] - sigma.shape[1]))))
        # if sigma.shape[0] < U.shape[1]:
            # sigma = np.vstack((sigma, np.zeros((U.shape[1] - sigma.shape[0], sigma.shape[1]))))        
        # print(np.dot(np.dot(U, sigma), V))
        # END YOUR CODE
        assert vecs.shape == (D, D)
        K = min(N, D)
        assert vals.shape == (K,)

        return vecs, vals

    def transform(self, X, n_components):
        """Center and project X onto a lower dimensional space using self.W_pca.

        Args:
            X: numpy array of shape (N, D). Each row is an example with D features.
            n_components: number of principal components..

        Returns:
            X_proj: numpy array of shape (N, n_components).
        """
        N, _ = X.shape
        X_proj = None
        # YOUR CODE HERE
        # We need to modify X in two steps:
        #     1. first substract the mean stored during `fit`
        #     2. then project onto a subspace of dimension `n_components` using `self.W_pca`
        X_center = X - self.mean
        X_proj = np.dot(X_center, self.W_pca[:, :n_components])                
        # END YOUR CODE

        assert X_proj.shape == (N, n_components), "X_proj doesn't have the right shape"

        return X_proj

    def reconstruct(self, X_proj):
        """Do the exact opposite of method `transform`: try to reconstruct the original features.

        Given the X_proj of shape (N, n_components) obtained from the output of `transform`,
        we try to reconstruct the original X.

        Args:
            X_proj: numpy array of shape (N, n_components). Each row is an example with D features.

        Returns:
            X: numpy array of shape (N, D).
        """
        N, n_components = X_proj.shape
        X = None

        # YOUR CODE HERE
        # Steps:
        #     1. project back onto the original space of dimension D
        #     2. add the mean that we substracted in `transform`
        # U, sigma, V = scipy.linalg.svd(self.W_pca[:, :n_components]) #通过svd对宽高不相等的矩阵求逆
        # #手动对sigma求逆，因为是对角阵所以只需对角线上元素取倒数，在封装好的函数里可能会被当做普通矩阵处理，为了节省时间手动求逆
        # sigma[np.abs(sigma)<1e-5] = 1e-5 #防止取倒数时爆炸 
        # sigma = 1 / sigma
        # sigma = np.diag(sigma)
        # sigma = np.hstack((sigma, np.zeros((sigma.shape[0], U.shape[0] - sigma.shape[1]))))
        # invert = np.dot(np.dot(V.T, sigma), U.T)  #公式上是V * S-1 * U.T，且svd函数算出来的V是V.T，所以语句中的V就变成了V.T
        # X_center = np.dot(X_proj, invert)        
        # X = X_center + self.mean
        X=X_proj.dot((self.W_pca[:,:n_components]).T)
        X=X+self.mean
        # END YOUR CODE

        return X


class LDA(object):
    """Class implementing Principal component analysis (PCA).

    Steps to perform PCA on a matrix of features X:
        1. Fit the training data using method `fit` (either with eigen decomposition of SVD)
        2. Project X into a lower dimensional space using method `transform`
        3. Optionally reconstruct the original X (as best as possible) using method `reconstruct`
    """

    def __init__(self):
        self.W_lda = None

    def fit(self, X, y):
        """Fit the training data `X` using the labels `y`.

        Will store the projection matrix in `self.W_lda`.

        Args:
            X: numpy array of shape (N, D). Each of the N rows represent a data point.
               Each data point contains D features.
            y: numpy array of shape (N,) containing labels of examples in X
        """
        N, D = X.shape

        scatter_between = self._between_class_scatter(X, y)
        scatter_within = self._within_class_scatter(X, y)

        e_vecs = None

        # YOUR CODE HERE
        # Solve generalized eigenvalue problem for matrices `scatter_between` and `scatter_within`
        # Use `scipy.linalg.eig` instead of numpy's eigenvalue solver.
        # Don't forget to sort the values and vectors in descending order.
        inv = np.linalg.inv(scatter_within)
        e_vals, e_vecs = np.linalg.eig(inv.dot(scatter_between))
        index = np.flipud(np.argsort(e_vals))
        e_vecs = e_vecs[:, index]
        # X = X.copy()
        # mu = X.mean(axis=0)
        # for i in np.unique(y):
            # mu_i = np.mean(X[y==i, :])
            # X[y==i, :] = X[y==i, :] - mu
        # print(scatter_within.dtype)
        # optim_W = np.linalg.inv(scatter_within) * X
        # e_vals, e_vecs = scipy.linalg.eig(optim_W)
        # END YOUR CODE

        self.W_lda = e_vecs

        # Check that the shape of `self.W_lda` is correct
        assert self.W_lda.shape == (D, D)

        # Each column of `self.W_lda` should have norm 1 (each one is an eigenvector)
        for i in range(D):
            assert np.allclose(np.linalg.norm(self.W_lda[:, i]), 1.0)

    def _within_class_scatter(self, X, y):
        """Compute the covariance matrix of each class, and sum over the classes.

        For every label i, we have:
            - X_i: matrix of examples with labels i
            - S_i: covariance matrix of X_i (per class covariance matrix for class i)
        The formula for covariance matrix is: X_centered^T X_centered
            where X_centered is the matrix X with mean 0 for each feature.

        Our result `scatter_within` is the sum of all the `S_i`

        Args:
            X: numpy array of shape (N, D) containing N examples with D features each
            y: numpy array of shape (N,), labels of examples in X

        Returns:
            scatter_within: numpy array of shape (D, D), sum of covariance matrices of each label
        """
        _, D = X.shape
        assert X.shape[0] == y.shape[0]
        scatter_within = np.zeros((D, D))

        for i in np.unique(y):
            # YOUR CODE HERE
            # Get the covariance matrix for class i, and add it to scatter_within 注意了，是add，对于每一类的数据单独求其协方差矩阵，得到的都是DxD矩阵，然后再相加起来，这种方法跟把所有数据一起来算协方差的结果是不一样的            
            scatter_within += np.cov(X[y==i, :].T).astype(np.float64) #np.cov函数是把每一行作为一个特征求协方差矩阵，我们要的是对每一列求，所以要先转置一下。又协方差矩阵是对称的，所以不用对结果再进行转置
            # 得到的Sw是个DxD矩阵，其第i行第j列的元素的含义是，样本点中每类样本点内部的数据中，第i个维度特征与第j个维度特征的协方差（与有几类标签无关）
            # END YOUR CODE

        return scatter_within

    def _between_class_scatter(self, X, y):
        """Compute the covariance matrix as if each class is at its mean.

        For every label i, we have:
            - X_i: matrix of examples with labels i
            - mu_i: mean of X_i.

        Our result `scatter_between` is the covariance matrix of X where we replaced every
        example labeled i with mu_i.

        Args:
            X: numpy array of shape (N, D) containing N examples with D features each
            y: numpy array of shape (N,), labels of examples in X

        Returns:
            scatter_between: numpy array of shape (D, D)
        """
        _, D = X.shape
        assert X.shape[0] == y.shape[0]
        scatter_between = np.zeros((D, D))
        X = X.copy() #这句是我自己加的
        mu = X.mean(axis=0)
        for i in np.unique(y):
            # YOUR CODE HERE
            X_i = X[y==i]
            mu_i = np.mean(X_i,axis=0)            
            N_i = X_i.shape[0]
            # X[y==i, :] = mu_i #把同类样本点的每一行替换成该类均值，然后把整个矩阵全部拿去计算，相当于把公式中的Ni也考虑了进去，就不需要再对矩阵做什么加权叠加了
            scatter_between += N_i * np.cov((mu_i-mu).T).astype(np.float64)
            # END YOUR CODE

        return scatter_between

    def transform(self, X, n_components):
        """Project X onto a lower dimensional space using self.W_pca.

        Args:
            X: numpy array of shape (N, D). Each row is an example with D features.
            n_components: number of principal components..

        Returns:
            X_proj: numpy array of shape (N, n_components).
        """
        N, _ = X.shape
        X_proj = None
        # YOUR CODE HERE
        # project onto a subspace of dimension `n_components` using `self.W_lda`
        X_proj = np.dot(X, self.W_lda[:, :n_components])
        # END YOUR CODE

        assert X_proj.shape == (N, n_components), "X_proj doesn't have the right shape"

        return X_proj
