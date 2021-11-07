import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

MAIN_IMAGE_PATH = 'Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg'
FIRST_FRAGMENT_PATH = 'Michelangelo/frag_eroded/frag_eroded_0.png'

# Method 1 : Harris Corner subpixel,
# https://docs.opencv.org/4.x/df/d54/tutorial_py_features_meaning.html
# Further refined pixel accuracy


def harris_subpixel(img_path):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray, np.float32(
        centroids), (5, 5), (-1, -1), criteria)
    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]
    img[res[:, 3], res[:, 2]] = [0, 255, 0]
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


main_image = harris_subpixel(MAIN_IMAGE_PATH)
fragment_image = harris_subpixel(FIRST_FRAGMENT_PATH)
plt.figure('Harris Corner subpixel Detector')
plt.subplot(2, 1, 1)
plt.title("Harris Corner subpixel Detector")
plt.imshow(main_image)
plt.subplot(2, 1, 2)
plt.imshow(fragment_image)
plt.show()

# Note : Green are refined corners, red are harris detected corners
#        rotation-invariant
#        not scale invariant
