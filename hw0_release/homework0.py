from __future__ import print_function
import random
import numpy as np
from linalg import *
from imageManip import *
import matplotlib.pyplot as plt


# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2
# %reload_ext autoreload

# M=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# # a=np.array([[1, 1, 0], [1, 1, 0]])
# # b=np.array([[-1, 2, 5], [-1, 2, 5]]).T
# a = np.array([1, 1, 0])
# b=np.array([-1, 2, 5]).T
# # print("M = \n", M)
# # print("a = ", a)
# # print("b = ", b)
# # print(len(b.shape))
#
# #Question 1.2
# # aDotB = dot_product(a, b)
# # print (aDotB)
#
# # ans = matrix_mult(M, a, b)
# # print (ans)
#
# # M = np.zeros([500, 500])
# # for i in range(M.shape[0]):
# #     for j in range(M.shape[1]):
# #         M[i][j] = random.random()
# M=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# u, s, v = gen_inv(M)
# s = sorted(s, reverse = True)
# # print(get_singular_values(M, 1))
# # print(get_singular_values(M, 2))
image1_path = 'image1.jpg'
image2_path = 'image2.jpg'
image1 = load(image1_path)
image2 = load(image2_path)

# display(image1)
# display(image2)

# new_image = change_value(image1)
# display(new_image)
# grey_image = convert_to_grey_scale(image1)
# display(grey_image)

# without_red = rgb_decomposition(image1, 'R')
# without_blue = rgb_decomposition(image1, 'B')
# without_green = rgb_decomposition(image1, 'G')
#
# display(without_red)
# display(without_blue)
# display(without_green)

# image_l = lab_decomposition(image1, 'L')
# image_a = lab_decomposition(image1, 'A')
# image_b = lab_decomposition(image1, 'B')
#
# display(image_l)
# display(image_a)
# display(image_b)

# image_h = hsv_decomposition(image1, 'H')
# image_s = hsv_decomposition(image1, 'S')
# image_v = hsv_decomposition(image1, 'V')
#
# display(image_h)
# display(image_s)
# display(image_v)

image_mixed = mix_images(image1, image2, channel1='R', channel2='G')
display(image_mixed)