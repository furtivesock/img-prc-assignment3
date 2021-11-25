import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

MAIN_IMAGE_PATH = 'Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg'
FIRST_FRAGMENT_PATH = 'Michelangelo/frag_eroded/frag_eroded_0.png'

# Method 4 : SURF (2006)
# https://docs.opencv.org/4.x/df/dd2/tutorial_py_surf_intro.html
# Faster than SIFT, but can't be tested because it is't included in OpenCV


def find_interest_points(img_path, nb_points=None):
    img = cv.imread(img_path)
    surf = cv.xfeatures2d.SURF_create(nb_points)
    kp, des = surf.detectAndCompute(img, None)
    # Here kp will be a list of keypoints
    # and des is a numpy array of shape (Number of Keypoints)×128

    # kp = sift.detect(img, None)
    print("Number of keypoints Detected: ", len(kp))

    img = cv.drawKeypoints(img, kp, img, color=(0, 0, 255))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


main_image = find_interest_points(MAIN_IMAGE_PATH)
fragment_image = find_interest_points(FIRST_FRAGMENT_PATH, 10)
plt.figure('SIFT')
plt.subplot(2, 1, 1)
plt.title("SIFT")
plt.imshow(main_image)
plt.subplot(2, 1, 2)
plt.imshow(fragment_image)
plt.show()

# Note : rotation-invariant
#        scale invariant
