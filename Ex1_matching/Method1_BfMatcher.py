import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MAIN_IMAGE_PATH = 'Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg'
FRAGMENT_PATH = 'Michelangelo/frag_eroded/frag_eroded_20.png'

img1 = cv.imread(FRAGMENT_PATH, cv.IMREAD_GRAYSCALE)    # queryImage
img2 = cv.imread(MAIN_IMAGE_PATH, cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate ORB detector
orb = cv.ORB_create(30000)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

img1 = cv.drawKeypoints(img1, kp1, None, color=(155, 0, 0), flags=0)
img2 = cv.drawKeypoints(img2, kp2, None, color=(155, 0, 0), flags=0)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1, kp1, img2, kp2,
                      matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.title("ORB + BFMatcher")
plt.imshow(img3), plt.show()
