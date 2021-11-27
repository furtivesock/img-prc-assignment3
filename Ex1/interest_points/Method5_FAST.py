from matplotlib import pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

MAIN_IMAGE_PATH = 'Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg'
FIRST_FRAGMENT_PATH = 'Michelangelo/frag_eroded/frag_eroded_20.png'

# Method 5 : FAST (2010)  (Features from Accelerated Segment Test)
# https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html
# It is several times faster than other existing corner detectors.
# But it is not robust to high levels of noise. It is dependent on a threshold.

# Note : rotation-invariant
#        scale invariant


def detect_corners(img_path, nb_points=None):
    img = cv.imread(img_path)
    fast = cv.FastFeatureDetector_create()
    kp = fast.detect(img, None)

    img2 = cv.drawKeypoints(img, kp, None, color=(0, 0, 255))
    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    return img2


main_image = detect_corners(MAIN_IMAGE_PATH)
fragment_image = detect_corners(FIRST_FRAGMENT_PATH, 10)
plt.figure('Fast')
plt.subplot(2, 1, 1)
plt.title("Fast")
plt.imshow(main_image)
plt.subplot(2, 1, 2)
plt.imshow(fragment_image)
plt.show()
