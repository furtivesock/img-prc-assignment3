from matplotlib import pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

MAIN_IMAGE_PATH = 'Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg'
FIRST_FRAGMENT_PATH = 'Michelangelo/frag_eroded/frag_eroded_20.png'

# Method 6 : ORB (2011)
# https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
# a fusion of FAST keypoint detector and BRIEF descriptor with performance boost

# Note : SIFT and SURF are patented but ORB is not


def detect_corners(img_path, nb_points=None):
    img = cv.imread(img_path, 0)
    # Initiate ORB detector
    orb = cv.ORB_create(nb_points)
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    print("Number of keypoints Detected: ", len(kp))
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0), flags=0)
    return img2


main_image = detect_corners(MAIN_IMAGE_PATH, 20000)
fragment_image = detect_corners(FIRST_FRAGMENT_PATH, 30)
plt.figure('ORB')
plt.subplot(2, 1, 1)
plt.title("ORB")
plt.imshow(main_image)
plt.subplot(2, 1, 2)
plt.imshow(fragment_image)
plt.show()
