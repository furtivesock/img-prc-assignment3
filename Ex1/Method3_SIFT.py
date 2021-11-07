import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

MAIN_IMAGE_PATH = 'Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg'
FIRST_FRAGMENT_PATH = 'Michelangelo/frag_eroded/frag_eroded_0.png'

# Method 3 : SIFT (2004)
# https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
# Changement in the scoring method


def Shi_Tomasi(img_path):
    img = cv.imread(img_path)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    # Here kp will be a list of keypoints
    # and des is a numpy array of shape (Number of Keypoints)×128

    # kp = sift.detect(img, None)
    print("Number of keypoints Detected: ", len(kp))

    img = cv.drawKeypoints(img, kp, img, color=(0, 0, 255))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


main_image = Shi_Tomasi(MAIN_IMAGE_PATH)
fragment_image = Shi_Tomasi(FIRST_FRAGMENT_PATH)
plt.figure('SIFT')
plt.subplot(2, 1, 1)
plt.title("SIFT")
plt.imshow(main_image)
plt.subplot(2, 1, 2)
plt.imshow(fragment_image)
plt.show()

# Note : rotation-invariant
#        scale invariant
