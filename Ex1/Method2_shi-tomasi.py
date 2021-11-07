import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

MAIN_IMAGE_PATH = 'Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg'
FIRST_FRAGMENT_PATH = 'Michelangelo/frag_eroded/frag_eroded_0.png'

# Method 2 : Shi-Tomasi Corner Detector & Good Features to Track (1994)
# https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html
# Changement in the scoring method


def Shi_Tomasi(img_path, intersest_point_number):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, intersest_point_number, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 5, 255, -1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


main_image = Shi_Tomasi(MAIN_IMAGE_PATH, 2000)
fragment_image = Shi_Tomasi(FIRST_FRAGMENT_PATH, 20)
plt.figure('Shi-Tomasi Corner Detector')
plt.subplot(2, 1, 1)
plt.title("Shi-Tomasi Corner Detector")
plt.imshow(main_image)
plt.subplot(2, 1, 2)
plt.imshow(fragment_image)
plt.show()

# Note : fixed number of interest points
#        rotation-invariant
#        not scale invariant
