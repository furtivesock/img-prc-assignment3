import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

MAIN_IMAGE_PATH = 'Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg'
FIRST_FRAGMENT_PATH = 'Michelangelo/frag_eroded/frag_eroded_0.png'

main_image = cv.imread(MAIN_IMAGE_PATH)
main_image_gray = cv.cvtColor(main_image, cv.COLOR_BGR2GRAY)
main_image_gray = np.float32(main_image_gray)

fragment_image = cv.imread(FIRST_FRAGMENT_PATH)
fragment_image_gray = cv.cvtColor(fragment_image, cv.COLOR_BGR2GRAY)
fragment_image_gray = np.float32(fragment_image_gray)

# Method 0 : cv.matchTemplate
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
# Does not work because of the framgent rotation

# Method 1 : Harris Corner and Edge Detector in 1988,
# https://docs.opencv.org/4.x/df/d54/tutorial_py_features_meaning.html
# It basically finds the difference in intensity for a displacement
# of (u,v) in all directions

dst = cv.cornerHarris(main_image_gray, 2, 3, 0.04)
dst_fragment = cv.cornerHarris(fragment_image_gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)
dst_fragment = cv.dilate(dst_fragment, None)

# Threshold for an optimal value, it may vary depending on the image.
main_image[dst > 0.01 * dst.max()] = [0, 0, 255]
main_image = cv.cvtColor(main_image, cv.COLOR_BGR2RGB)
fragment_image[dst_fragment > 0.01 * dst_fragment.max()] = [0, 0, 255]
fragment_image = cv.cvtColor(fragment_image, cv.COLOR_BGR2RGB)

plt.figure('Harris Corner and Edge Detector')
plt.subplot(2, 1, 1)
plt.title("Harris Corner and Edge Detector")
plt.imshow(main_image)
plt.subplot(2, 1, 2)
plt.imshow(fragment_image)
plt.show()

# Note : rotation-invariant
#        not scale invariant
