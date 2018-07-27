import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io
import copy


def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = io.imread(image_path)

    return out


def display(img):
    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = 0.5 * np.power(image,2)

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = (image[:, : ,0] * 0.299 + image[:, : ,1] * 0.587 + image[:, : ,2] * 0.114)

    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = copy.deepcopy(image)

    if channel == 'R':
        out[:, :, 0] = 0
    elif channel == 'G':
        out[:, :, 1] = 0
    elif channel == 'B':
        out[:, :, 2] = 0

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    if channel == 'L':
        return lab[:, :, 0]
    elif channel == 'A':
        return lab[:, :, 1]
    elif channel == 'B':
        return lab[:, :, 2]



def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    if channel == 'H':
        return hsv[:, :, 0]
    elif channel == 'S':
        return hsv[:, :, 1]
    elif channel == 'V':
        return hsv[:, :, 2]


def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    m, n, p= image1.shape
    out = copy.deepcopy(image1)
    if channel1 == 'R':
        out[:, :, 0] = 0
    elif channel1 == 'G':
        out[:, :, 1] = 0
    elif channel1 == 'B':
        out[:, :, 2] = 0

    out[:, round(m/2):, :] = image2[:, round(m/2):, :]
    if channel2 == 'R':
        out[:, round(m/2):, 0]= 0
    elif channel2 == 'G':
        out[:, round(m/2):, 1] = 0
    elif channel2 == 'B':
        out[:, round(m/2):, 2] = 0

    return out
