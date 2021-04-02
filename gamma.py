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
import cv2
from ex1_utils import LOAD_GRAY_SCALE
import numpy as np

im = None


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global im
    im = cv2.imread(img_path).astype(np.float32)  # read image

    # create GUI
    cv2.namedWindow("gamma correction", flags=cv2.WINDOW_NORMAL)
    cv2.createTrackbar('gammaX100', 'gamma correction', 0, 200, tracker_act)
    cv2.waitKey()


def tracker_act(val):
    p_im = (im/255)**(val/100)  # the gamma correction
    cv2.imshow("gamma correction", p_im)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
