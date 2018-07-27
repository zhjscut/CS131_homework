import numpy as np
from skimage import feature,data, color, exposure, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import interpolation
import math

def hog_feature(image, pixel_per_cell = 8):
    ''' 
    Compute hog feature for a given image.
    
    Hint: use the hog function provided by skimage
    
    Args:
        image: an image with object that we want to detect
        pixel_per_cell: number of pixels in each cell, an argument for hog descriptor
        
    Returns:
        hogFeature: a vector of hog representation
        hogImage: an image representation of hog provided by skimage
    '''
    ### YOUR CODE HERE
    hogFeature, hogImage = feature.hog(image, pixels_per_cell=(pixel_per_cell, pixel_per_cell), visualise=True)
    ### END YOUR CODE
    return (hogFeature, hogImage)

def sliding_window(image, base_score, stepSize, windowSize, pixel_per_cell=8):
    ''' A sliding window that checks each different location in the image, 
        and finds which location has the highest hog score. The hog score is computed
        as the dot product between hog feature of the sliding window and the hog feature
        of the template. It generates a response map where each location of the
        response map is a corresponding score. And you will need to resize the response map
        so that it has the same shape as the image.
    
    Args:
        image: an np array of size (h,w)
        base_score: hog representation of the object you want to find, an array of size (m,)
        stepSize: an int of the step size to move the window
        windowSize: a pair of ints that is the height and width of the window
    Returns:
        max_score: float of the highest hog score 
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        response_map: an np array of size (h,w)
    '''
    # slide a window across the image
    (max_score, maxr, maxc) = (0,0,0)
    winH, winW = windowSize
    H,W = image.shape
    pad_image = np.lib.pad(image, ((winH//2,winH-winH//2),(winW//2, winW-winW//2)), mode='constant')
    response_map = np.zeros((H//stepSize+1, W//stepSize+1))
    
    ### YOUR CODE HERE
    # sum_hog = np.sum(base_score) #用于做相关时去除能量的影响
    for i in range(0, response_map.shape[0]): #考虑到H和W不一定是winH和winW的整数倍，是否可以采取这样的措施：循环只到倒数第二行/列，最后一行/列单独处理，最后剩下多大就跟多大的face_feature做相关，最后记得均衡一下（得到的响应值/最后一块的大小*正常块的大小）
        for j in range(0, response_map.shape[1]):
            image_batch = pad_image[i*stepSize : i*stepSize+winH, j*stepSize : j*stepSize+winW]
            (hogFeature, _) = hog_feature(image_batch, pixel_per_cell)
            response_map[i, j] = np.sum(base_score * hogFeature) #/ (sum_hog * np.sum(hogFeature)) #之前还在想用不用去除能量的影响，以为一张白纸跟人脸做相关会是最大，后来发现不对，做相关的不是原像素而是hog特征，而白纸的hog特征是全零，因此不会是最大。而且去除能量的效果并不好，得到了错误的识别结果，综上，不需要去除能量的影响
            # 这样得到的位置似乎更精确一点
            # if score>max_score:
            #     max_score=score
            #     maxr=i*stepSize-winH//2
            #     maxc=j*stepSize-winW//2
    max_index, max_score = np.argmax(response_map), np.max(response_map)
    maxr, maxc = int(max_index / response_map.shape[1] - np.ceil(winH//2/stepSize)), int(max_index % response_map.shape[1] - np.ceil(winW//2/stepSize)) #因为直接argmax出来的下标是与填充后的图像相对应的下标，而输出要的是对应与原图片，所以需要做一下校正
    # max_score = response_map[maxr, maxc] #不能用这个语句来求解最大值，一开始没有校正时是可以的，加了校正之后的位置就是不是原来的响应图的最大值了
    maxr, maxc = maxr * int(stepSize), maxc * int(stepSize) #将response_map以及最大值下标转换成原图大小
    response_map = resize(response_map,(H,W),preserve_range=1)    
    ### END YOUR CODE
    
    
    return (max_score, maxr, maxc, response_map)


def pyramid(image, scale=0.9, minSize=(200, 100)):
    '''
    Generate image pyramid using the given image and scale.
    Reducing the size of the image until on of the height or
    width reaches the minimum limit. In the ith iteration, 
    the image is resized to scale^i of the original image.
    
    Args:
        image: np array of (h,w), an image to scale
        scale: float of how much to rescale the image each time
        minSize: pair of ints showing the minimum height and width
        
    Returns:
        images: a list containing pair of 
            (the current scale of the image, resized image)
    '''
    # yield the original image
    images = []
    current_scale = 1.0
    images.append((current_scale, image))
    # keep looping over the pyramid
    ### YOUR CODE HERE
    while 1:
        image = rescale(image, scale)
        current_scale *= scale
        if image.shape[0] < minSize[0] or image.shape[1] < minSize[1]: #小于图片大小的下限时退出循环，不保存低于下限的那张图片
            break 
        images.append((current_scale, image))
    ### END YOUR CODE
    return images

def pyramid_score(image, base_score, shape, stepSize=20, scale = 0.9, pixel_per_cell = 8):
    '''
    Calculate the maximum score found in the image pyramid using sliding window.
    
    Args:
        image: np array of (h,w)
        base_score: the hog representation of the object you want to detect
        shape: shape of window you want to use for the sliding_window
        
    Returns:
        max_score: float of the highest hog score 
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        max_scale: float of scale when the max_score is found
        max_response_map: np array of the response map when max_score is found
    '''
    max_score = 0
    maxr = 0
    maxc = 0
    max_scale = 1.0
    max_response_map =np.zeros(image.shape)
    images = pyramid(image, scale)

    ### YOUR CODE HERE
    for result in images:
        (scale, image) = result
        (score, r, c, response_map) = sliding_window(image, base_score, stepSize, shape, pixel_per_cell)
        print(score)
        if score > max_score:
            max_score = score
            maxr = r
            maxc = c
            max_scale = scale
            max_response_map = response_map
    ### END YOUR CODE
    return max_score, maxr, maxc, max_scale, max_response_map


def compute_displacement(part_centers, face_shape):
    ''' Calculate the mu and sigma for each part. d is the array 
        where each row is the main center (face center) minus the 
        part center. Since in our dataset, the face is the full
        image, face center could be computed by finding the center
        of the image. Vector mu is computed by taking an average from
        the rows of d. And sigma is the standard deviation among 
        among the rows. Note that the heatmap pixels will be shifted 
        by an int, so mu is an int vector.
    
    Args:
        part_centers: np array of shape (n,2) containing centers 
            of one part in each image
        face_shape: (h,w) that indicates the shape of a face
    Returns:
        mu: (1,2) vector
        sigma: (1,2) vector
        
    '''
    d = np.zeros((part_centers.shape[0],2))
    ### YOUR CODE HERE
    face_center = np.array([(face_shape[0]-1)/2, (face_shape[1]-1)/2]) #在该样本集中，脸部占满了整个图像，图像的中心即为脸部的中心。注意坐标是从0到n-1
    d = face_center - part_centers
    mu = np.round(np.mean(d, axis=0)).astype(np.int)
    sigma = np.std(d, axis=0)
    ### END YOUR CODE
    return mu, sigma
        
def shift_heatmap(heatmap, mu):
    '''First normalize the heatmap to make sure that all the values 
        are not larger than 1.
        Then shift the heatmap based on the vector mu.

        Args:
            heatmap: np array of (h,w)
            mu: vector array of (1,2)
        Returns:
            new_heatmap: np array of (h,w)
    '''
    ### YOUR CODE HERE
    heatmap = heatmap / np.max(heatmap) #暂时关掉，后面要开回来
    h = heatmap
    dx, dy = mu[0], mu[1]
    assert abs(dx) <= heatmap.shape[0] and abs(dy) <= heatmap.shape[1], 'wrong input mu'
    # 圆周移位。原以为要分情况讨论，最后发现mu在4个象限对应的式子的形式一模一样
    new_heatmap = np.vstack(( np.hstack((h[-dx:, -dy:], h[-dx:, 0:-dy])), np.hstack((h[0:-dx, -dy:], h[0:-dx, 0:-dy])) ))
    # if dx >= 0 and dy >= 0:
    #     new_heatmap = np.vstack(( np.hstack((h[-dx:, -dy:], h[-dx:, 0:-dy])), np.hstack((h[0:-dx, -dy:], h[0:-dx, 0:-dy])) ))
    # elif dx >= 0 and dy < 0:
    #     new_heatmap = np.vstack(( np.hstack((h[-dx:, -dy:], h[-dx:, 0:-dy])), np.hstack((h[0:-dx, -dy:], h[0:-dx, 0:-dy])) ))
    # elif dx < 0 and dy >= 0:
    #     new_heatmap = np.vstack(( np.hstack((h[-dx:, -dy:], h[-dx:, 0:-dy])), np.hstack((h[0:-dx, -dy:], h[0:-dx, 0:-dy])) ))
    # elif dx < 0 and dy < 0:
    #     new_heatmap = np.vstack(( np.hstack((h[-dx:, -dy:], h[-dx:, 0:-dy])), np.hstack((h[0:-dx, -dy:], h[0:-dx, 0:-dy])) ))

        
    ### END YOUR CODE
    return new_heatmap
    

def gaussian_heatmap(heatmap_face, heatmaps, sigmas):
    '''
    Apply gaussian filter with the given sigmas to the corresponding heatmap.
    Then add the filtered heatmaps together with the face heatmap.
    Find the index where the maximum value in the heatmap is found. 
    
    Hint: use gaussian function provided by skimage
    
    Args:
        image: np array of (h,w)
        sigma: sigma for the gaussian filter
    Return:
        new_image: an image np array of (h,w) after gaussian convoluted
    '''
    ### YOUR CODE HERE
    heatmap = np.zeros(heatmap_face.shape)
    # 各个热力图已在shift函数中被归一化过，此处无需重复归一化
    for i in range(len(heatmaps)):
        heatmap_i = heatmaps[i]
        sigma = sigmas[i]
        heatmap += gaussian(heatmap_i, sigma)
    heatmap += heatmap_face
    index = np.argmax(heatmap)
    r,c = index // heatmap.shape[1], index % heatmap.shape[1]
    ### END YOUR CODE
    return heatmap, r , c

      
def detect_multiple(image, response_map):
    '''
    Extra credit
    '''
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return detected_faces

            

    