import numpy as np
from skimage import color
from skimage import filters
import copy


def energy_function(image): #对它的计算方法不是很理解，按零延拓正常计算边界不好吗？
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: you can use np.gradient here

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W, _ = image.shape
    image_gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    G_x, G_y = np.gradient(image_gray)
    out = np.abs(G_x) + np.abs(G_y)
    # out = np.zeros((H, W))
    # for i in range(1, H-1):
        # for j in range(1, W-1):
            # out[i, j] = abs(image_gray[i+1, j] - image_gray[i-1, j]) / 2 + abs(image_gray[i, j+1] - image_gray[i, j-1]) / 2
    # for i in range(1, H-1):
        # out[i, 0] = 0.5 * abs(image_gray[i-1, 0] - image_gray[i+1, 0]) + abs(image_gray[i, 0] - image_gray[i, 1])
        # out[i, W-1] = 0.5 * abs(image_gray[i-1, W-1] - image_gray[i+1, W-1]) + abs(image_gray[i, W-1] - image_gray[i, W-2])
    # for i in range(1, W-1):
        # out[0, i] = 0.5 * abs(image_gray[0, i-1] - image_gray[0, i+1]) + abs(image_gray[0, i] - image_gray[1, i])
        # out[H-1, i] = 0.5 * abs(image_gray[H-1, i-1] - image_gray[H-1, i+1]) + abs(image_gray[H-1, i] - image_gray[H-2, i])
    
    # out[0, 0] = abs(image_gray[1, 0] - image_gray[0, 0]) + abs(image_gray[0, 1] - image_gray[0, 0])
    # out[0, W-1] = abs(image_gray[0, W-2] - image_gray[0, W-1]) + abs(image_gray[1, W-1] - image_gray[0, W-1])
    # out[H-1, 0] = abs(image_gray[H-2, 0] - image_gray[H-1, 0]) + abs(image_gray[H-1, 1] - image_gray[H-1, 0])
    # out[H-1, W-1] = abs(image_gray[H-2, W-1] - image_gray[H-1, W-1]) + abs(image_gray[H-1, W-2] - image_gray[H-1, W-1])
    
    return out


def compute_cost(image, energy, axis=1):
    """Computes optimal cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Make sure your code is vectorized because this function will be called a lot.
    You should only have one loop iterating through the rows.

    Args:
        image: not used for this function
               (this is to have a common interface with compute_forward_cost)
        energy: numpy array of shape (H, W)
        axis: compute cost in width (axis=1) or height (axis=0)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """
    #-1代表从左边（上边）过来，0代表从正上方（正左方）过来，1代表从右边（下边）过来
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths    
    for i in range(1, H):
        center = cost[i-1, :]
        left = np.hstack(([100000.], cost[i-1, 0: -1]))
        right = np.hstack(([cost[i-1, 1:], [100000.]]))
        temp = np.vstack((left, center, right))
        cost[i, :] = energy[i, :] + np.min(temp, 0)
        paths[i, :] = np.argmin(temp, axis=0) - 1 #十分巧妙地将左中右与-1 0 1对应了起来
    # mini_energy = np.zeros(W)
    # for i in range(1, H):    
        # mini_energy[0] = min(cost[i-1, 0], cost[i-1, 1])
        # if cost[i-1, 0] < cost[i-1, 1]:
            # paths[i, 0] = 0
        # else:
            # paths[i, 0] = 1
        # mini_energy[W-1] = min(cost[i-1, W-1], cost[i-1, W-2])
        # if cost[i-1, W-1] < cost[i-1, W-2]:
            # paths[i, W-1] = 0
        # else:
            # paths[i, W-1] = -1
            
        # for j in range(1, W-1):
            # mini_energy[j] = min(cost[i-1, j], cost[i-1, j-1], cost[i-1, j+1])
            # if mini_energy[j] == cost[i-1, j]:
                # paths[i, j] = 0
            # elif mini_energy[j] == cost[i-1, j-1]:
                # paths[i, j] = -1
            # else:
                # paths[i, j] = 1
        # cost[i] = mini_energy + energy[i]    

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    """Backtracks the paths map to find the seam ending at (H-1, end)

    To do that, we start at the bottom of the image on position (H-1, end), and we
    go up row by row by following the direction indicated by paths:
        - left (value -1)
        - middle (value 0)
        - right (value 1)

    Args:
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
        end: the seam ends at pixel (H, end)

    Returns:
        seam: np.array of indices of shape (H,). The path pixels are the (i, seam[i])
    """
    H, W = paths.shape
    seam = np.zeros(H, dtype=np.int)

    # Initialization
    seam[H-1] = end
    for i in range(H-2, -1, -1):
        seam[i] = seam[i+1] + paths[i+1, seam[i+1]]

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    """Remove a seam from the image.

    This function will be helpful for functions reduce and reduce_forward.

    Args:
        image: numpy array of shape (H, W, C) or shape (H, W)
        seam: numpy array of shape (H,) containing indices of the seam to remove

    Returns:
        out: numpy array of shape (H, W-1, C) or shape (H, W-1)
    """

    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)        

    # out = None
    out = np.zeros((image.shape[0], image.shape[1]-1, image.shape[2]))
    H, W, C = image.shape
    
    ### YOUR CODE HERE
    for i in range(0, H):
        out[i, :, :] = np.delete(image[i, :, :], seam[i], axis=0)
    ### END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process.

    At each step, we remove the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, 3)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, 3) if axis=0, or (H, size, 3) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    #axis = 1是去掉一列
    for i in range(0, W-size):
        energy = efunc(out)
        vcost, vpaths = cfunc(out, energy)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpaths, end)
        out = remove_seam(out, seam)
    ### END YOUR CODE
    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def duplicate_seam(image, seam):
    """Duplicates pixels of the seam, making the pixels on the seam path "twice larger".

    This function will be helpful in functions enlarge_naive and enlarge.

    Args:
        image: numpy array of shape (H, W, C)
        seam: numpy array of shape (H,) of indices

    Returns:
        out: numpy array of shape (H, W+1, C)
    """

    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    ### YOUR CODE HERE
    for i in range(0, H):
        out[i, :, :] = np.insert(image[i, :, :], seam[i], values=image[i, seam[i], :], axis=0) 
    ### END YOUR CODE

    return out


def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Increases the size of the image using the seam duplication process.

    At each step, we duplicate the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to increase height or width to (depending on axis)
        axis: increase in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    ### YOUR CODE HERE
    for i in range(0, size-W):
        energy = efunc(out)
        vcost, vpaths = cfunc(out, energy)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpaths, end)
        out = duplicate_seam(out, seam)
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    indices = np.tile(range(W), (H, 1))  # shape (H, W)
    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = backtrack_seam(paths, end)

        # Remove that seam from the image
        image = remove_seam(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]] == 0), \
            "we are overwriting seams" #保证找到的seam像素点之前没被找到过
        seams[np.arange(H), indices[np.arange(H), seam].astype(np.int)] = np.int(i + 1) #把这些像素点对应位置做上标记

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = remove_seam(indices, seam).astype(np.int) #indices随着remove的过程不断变瘦，但是它的元素值——seam的列号却一直保存了下来。更新seams看的不是indices的行号列号，而是里面的元素值！

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Enlarges the size of the image by duplicating the low energy seams.

    We start by getting the k seams to duplicate through function find_seams.
    We iterate through these seams and duplicate each one iteratively.

    Use functions:
        - find_seams
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: enlarge in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    ### YOUR CODE HERE
    indices = np.tile(range(W), (H, 1))  # shape (H, W)
    seams = find_seams(out, size - W)
    for i in range(0, size-W):
        index_seam = np.argwhere(seams == i+1)[:,1]
        seam = indices[np.arange(H), index_seam]
        out = duplicate_seam(out, seam)
        for j in range(0, H):
            indices[j, index_seam[j]:] += 1 #更新变化了的列号
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def compute_forward_cost(image, energy):
    """Computes forward cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    Make sure to add the forward cost introduced when we remove the pixel of the seam.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Args:
        image: numpy array of shape (H, W, 3) or (H, W)
        energy: numpy array of shape (H, W)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """
    #我理解的话 因为你energy是原图像的梯度和。但是你要删一个seam的话 会有原来碰不到的像素碰到一起。用旧的梯度没意义吧。用新碰到一起的像素值的差可以反映一种边界信息吧。差值越大说明在边界
    #公式中的P是能量的意思，图片里的p是像素的意思
    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if j > 0 and j < W - 1:
            cost[0, j] += np.abs(image[0, j+1] - image[0, j-1])
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    # print(image[:,1,np.newaxis].shape, image.shape, image[:, -2].shape)
    image_tmp = np.hstack((image[:, 1, np.newaxis], image, image[:, -2, np.newaxis]))
    # print(image_tmp.shape)
    for i in range(1, H):
        center = np.abs(image_tmp[i, :-2] - image_tmp[i, 2:]) + cost[i-1, :]
        # print(center.shape)
        # print(np.abs(image_tmp[i, : -2] - image_tmp[i, 2:]).shape, np.abs(image_tmp[i, : -2] - image_tmp[i-1, 1:-1]).shape, np.hstack(([100000.], cost[i-1, 0:-1])).shape)
        left = np.abs(image_tmp[i, : -2] - image_tmp[i, 2:]) + np.abs(image_tmp[i, : -2] - image_tmp[i-1, 1:-1]) + np.hstack(([100000.], cost[i-1, 0:-1]))
        right = np.abs(image_tmp[i, : -2] - image_tmp[i, 2:]) + np.abs(image_tmp[i-1, 1: -1] - image_tmp[i, 2:]) + np.hstack((cost[i-1, 1:], [100000.]))
        
        temp = np.vstack((left, center, right))
        cost[i, :] = energy[i, :] + np.min(temp, 0)
        paths[i, :] = np.argmin(temp, axis=0) - 1 
    ### END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Hint: do we really need to compute the whole cost map again at each iteration?

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    out = np.copy(image)

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
