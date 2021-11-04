import cv2 as cv
import os
import matplotlib.pyplot as plt


img = cv.imread(
    "Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

template = cv.imread("Michelangelo/frag_eroded/frag_eroded_0.png")
template = cv.cvtColor(template, cv.COLOR_BGR2RGB)


# plt.imshow(template)
# plt.show()
# plt.imshow(img)
# plt.show()

# All the matching methods that we are going to test
methods = ["cv.TM_CCOEFF", "cv.TM_CCOEFF_NORMED", "cv.TM_CCORR",
           "cv.TM_CCORR_NORMED", "cv.TM_SQDIFF", "cv.TM_SQDIFF_NORMED"]

for method in methods:
    print("evaluating method " + method)
    img_copy = img.copy()
    template_copy = template.copy()
    res = cv.matchTemplate(img_copy, template_copy, eval(method))
    # plt.imshow(res)
    # plt.show()

    # Find the result maximum value and its position
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    #  Depending on the method, the minimum value may be the match value
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    print(top_left)

    # Draw a rectangle around the result
    height, width, channels = template.shape
    bottom_right = (top_left[0] + width, top_left[1] + height)

    cv.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 6)

    plt.figure(method)
    plt.subplot(3, 1, 1)
    plt.imshow(template)
    plt.title("Fragment")
    plt.subplot(3, 1, 2)
    plt.imshow(res)
    plt.title("Fragment MATCHING MAP")
    plt.subplot(3, 1, 3)
    plt.imshow(img_copy)
    plt.title("Fragment DETECTION")
    plt.suptitle(method)
    OUTPUT_FOLDER = "out"
    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    cv.imwrite(f"{OUTPUT_FOLDER}/{method}.jpg", img_copy)

plt.show()
