"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import math
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 323918599


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    try:
        im = cv2.imread(filename).astype(np.float32)  # import the image
    except:
        return None

    if representation == 2:  # convert from BGR to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if representation == 1 and len(im.shape) == 3:  # if needed, convert from BGR to gray scale
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im = im / 255  # normalize to [0,1]

    return im


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    im = imReadAndConvert(filename, representation)  # read the image
    if representation == 1:
        plt.imshow(im, cmap='gray')  # display
    else:
        plt.imshow(im)  # display
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    rgb_to_yiq = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])

    return np.dot(imgRGB, rgb_to_yiq.T)

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    rgb_to_yiq = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
    yiq_to_rgb = np.linalg.inv(rgb_to_yiq)  # the inverse matrix

    return np.dot(imgYIQ, yiq_to_rgb.T)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """

    if len(imgOrig.shape) == 3:  # if the image is RGB
        yiq_im = transformRGB2YIQ(imgOrig)  # transform to yiq
        imgEq_y, histOrig, histEq = hsitogramEqualize_2D(yiq_im[:, :, 0])  # do histogram Equalization on Y channel
        imgEq = np.array([imgEq_y, yiq_im[:, :, 1], yiq_im[:, :, 2]])  # append the I and Q channels
        imgEq = np.swapaxes(np.swapaxes(imgEq, 0, 1), 1, 2)
        imgEq = transformYIQ2RGB(imgEq)

    else:
        imgEq, histOrig, histEq = hsitogramEqualize_2D(imgOrig)

    return imgEq, histOrig, histEq


def hsitogramEqualize_2D(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of a 2D-image
        :param imgOrig: Original Histogram
        :ret
    """

    imgOrig_255 = np.round(imgOrig * 255).astype(np.int)  # set the image to be in range [0,255]
    histOrig = np.histogram(imgOrig_255, bins=256)[0]  # generate the histogram

    # create the look up table
    cumSum = np.cumsum(histOrig)
    normCumSum = cumSum / sum(histOrig)
    lookUpTable = np.round(normCumSum * 255)

    # do the Equalization on the image and normalize back to [0,1]
    imgEq = (np.array(lookUpTable[imgOrig_255.flatten()]).reshape(imgOrig.shape)) / 255
    histEq = np.histogram(np.round(imgEq*255), bins=256)[0]

    return imgEq, histOrig, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    if nQuant > len(np.unique(imOrig)):  # make sure there is enough values
        return None, None

    if len(imOrig.shape) == 3:  # if the image is RGB

        yiq_im = transformRGB2YIQ(imOrig)  # convert to YIQ
        y_im = yiq_im[:, :, 0]

        qImages, errors = quantizeImage_2D(y_im, nQuant, nIter)  # do the quantization on the Y channel

        # re-append the I and Q channels, and convert back to RGB
        qImages_yiq = [np.swapaxes(np.swapaxes(np.array([imq, yiq_im[:, :, 1], yiq_im[:, :, 2]]), 0, 1), 1, 2) for imq in qImages]
        qImages = [transformYIQ2RGB(im) for im in qImages_yiq]

        return qImages, errors

    else:  # if the image is gray scale
        qImages, errors = quantizeImage_2D(imOrig, nQuant, nIter)  # run the quantization

        return qImages, errors


def quantizeImage_2D(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
     Quantized a 2D-image in to **nQuant** colors
        :param imOrig: The original image
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    imOrig = np.round(imOrig * 255).astype(np.int)  # map values to [0,255]

    # initiate the borders (Z) and the 'means' (Q)
    hist = np.histogram(imOrig, bins=256)[0]
    Z = np.array([0, *equal_sum_partition(hist, nQuant), 256])
    Q = np.array([(Z[i] + Z[i + 1]) // 2 for i in range(nQuant)])

    qImages = []
    errors = []

    n_pixels = sum(hist)

    for i in range(nIter):
        Z, Q = quantizeImage_iteration(hist, Z, Q)  # one iteration of the algorithm

        # map each pixel to the matching value
        LUT = np.concatenate([np.full(Z[i+1]-Z[i], Q[i]) for i in range(len(Q))])
        qIm = np.array(LUT)[imOrig.flatten()].reshape(imOrig.shape)
        # calculate the square error
        err = math.sqrt(sum(((imOrig - qIm) ** 2).flatten())) / n_pixels

        errors = errors + [err]
        qImages  = qImages + [qIm/255]

    return qImages, errors


def quantizeImage_iteration(hist: np.array, Z: np.array, Q: np.array) -> (np.array, np.array):
    """
    one iteration of the quantization algorithm
    :param hist: histogram of the original image
    :param Z: current borders vector
    :param Q: current 'means' vector
    :return: the new borders and 'means' vectors
    """

    # calculate the new 'means' vector
    split_hist = np.split(hist, Z[1:len(Z) - 1])
    split_indexes = np.split(np.arange(256), Z[1:len(Z) - 1])
    Q = np.array([np.dot(split_hist[i], split_indexes[i])//sum(split_hist[i]) for i in range(len(Q))])

    # calculate the new borders vector
    Z = np.concatenate(([0], ((Q[:len(Q)-1] + Q[1:])//2), [256]))
    return Z, Q

def fill_array(p: tuple) -> List[int]:
    print(p)
    return np.full(p[0], p[1])


def equal_sum_partition(arr: np.array, p: int):
    """
    divide the array into equal-sum parts
    :param arr:
    :param p: number of parts
    :return: the indexes for division
    """
    ac = np.cumsum(arr)
    cum_part_sums = np.array(range(1, p)) * (ac[-1] // p)  # generates the cumulative sums of each part
    return np.searchsorted(ac, cum_part_sums)  # finds the indexes where the cumulative sums are sandwiched
