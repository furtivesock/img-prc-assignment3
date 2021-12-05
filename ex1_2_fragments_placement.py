"""Exercise 1 (Pt. 2)
Try to place the fragments on the fresco using the first match, without filtering

Authors: Tom Mansion <tom.mansion@universite-paris-saclay.fr>, Sophie Nguyen <sophie.nguyen@universite-paris-saclay.fr>
"""

import os
import math
import cv2 as cv
import numpy as np

def get_all_fragments_path(folder_path):
    """
    Get all fragments from a folder.
    """
    folder_path += "/frag_eroded/"
    fragments = []
    for fragments_file in os.listdir(folder_path):
        if fragments_file.endswith(".png"):
            fragments.append(folder_path + fragments_file)
    return fragments

def get_ORB_interest_points(image, number_of_keypoints):
    """
    Get the ORB interest points from a fragment.
    """

    # Initiate ORB detector
    orb = cv.ORB_create(number_of_keypoints)

    # find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

def get_BFMatcher_associations(from_des, to_des, distance_threshold=0.75):
    """
    Get the BFMatcher associations.
    Args:
        from_des: descriptors of the fragment
        to_des: descriptors of the main image
    """
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(from_des, to_des, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < distance_threshold * n.distance:
            good_matches.append([m])

    return good_matches

def add_fragment_to_target_image(fragment_info, fragment_img, target_image, target_width, target_height) -> None:
    """Place a rotated fragment on the fresco
    by filling the background pixels on the right position with fragment ones

    Args:
        fragment_info (object):     Fragment settings: coordinates (x, y)
        fragment_img (2D array):    Image of a fragment
        target_image (2D array):    Original fresco background
        target_width (int):         Fresco width
        target_height (int):        Fresco height
    """
    fragment_coord_y = 0
    fragment_coord_x = 0

    # Iterate each one of the fragment pixels
    for row in fragment_img:
        # Calculate the y coodinate of the fragment pixel on the target image
        target_image_fragment_y = fragment_info["y"] - \
            int(fragment_img.shape[0] / 2) + fragment_coord_y
        # Check that we are the fragment image isn't going to be out of the image
        if target_image_fragment_y < 0 or target_image_fragment_y >= target_height:
            continue

        for pixel in row:
            # Because target image is in JPG (R, G, B) and fragment_image if alpha > 0 (non-transparent) --> add pixel to target_image
            if pixel[3] > 200:
                # The alpha is > 0 : the pixel isn't tranparent
                # Calculate the x coordinate of the fragment pixel on the target image
                target_image_fragment_x = fragment_info["x"] - int(
                    fragment_img.shape[1] / 2) + fragment_coord_x
                # Check that the fragment image isn't going to be out of the image
                if target_image_fragment_x < 0 or target_image_fragment_x >= target_width:
                    continue

                # Add the pixel to the target image
                target_image[target_image_fragment_y][target_image_fragment_x] = [
                    pixel[0], pixel[1], pixel[2]]

            fragment_coord_x += 1

        fragment_coord_y += 1
        fragment_coord_x = 0


# Main : test the methods
if __name__ == "__main__":

    # Get the fresco
    fresco = cv.imread(
        "Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg", cv.IMREAD_GRAYSCALE)
    result = cv.imread(
        "Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg", cv.IMREAD_UNCHANGED)

    # Set a light background with the original image
    # It will help us understand how the fragments match
    # Add a low alpha channel to the original_img
    ORIGINAL_IMAGE_OPACITY = 0.3
    target_height, target_width, _ = result.shape
    white_background = np.full(
        (target_height, target_width, 3), 255, dtype=np.uint8)
    background = cv.addWeighted(
        result, ORIGINAL_IMAGE_OPACITY, white_background, 1 - ORIGINAL_IMAGE_OPACITY, 0)

    # Get the ORB interest points
    fresco_kp, fresco_des = get_ORB_interest_points(
        fresco, number_of_keypoints=30000)

    print("fresco_kp: ", len(fresco_kp))

    # Get the fragments
    fragments = get_all_fragments_path("Michelangelo")
    print("Number of fragments : ", len(fragments))

    ok_frag_kp_nb = 0
    ok_frag_match_nb = 0
    for fragment_path in fragments:
        print("  ", fragment_path)

        # Load the fragment image
        fragment = cv.imread(fragment_path, cv.IMREAD_GRAYSCALE)

        # Get the ORB interest points
        fragment_kp, fragment_des = get_ORB_interest_points(
            fragment, number_of_keypoints=30000)
        print("   fragment_kp: ", len(fragment_kp))

        if len(fragment_kp) < 2:
            print("     ! Not enougth kp")
            continue

        ok_frag_kp_nb += 1

        # Get the BFMatcher associations
        matches = get_BFMatcher_associations(fragment_des, fresco_des, 0.75)
        print("   matches: ", len(matches))
        if len(matches) < 2:
            print("     ! Not enougth matches")
            continue

        ok_frag_match_nb += 1
        fragment_result = cv.imread(fragment_path, cv.IMREAD_UNCHANGED)

        # Calculating the fragment position
        fresco_match_point = fresco_kp[matches[0][0].trainIdx].pt
        fragment_match_point = fragment_kp[matches[0][0].queryIdx].pt
        fresco_match_point = (
            int(fresco_match_point[0]),
            int(fresco_match_point[1])
        )

        # Set the match point related to the center of the fragment
        fragment_match_point = (
            int(fragment_match_point[0] - fragment.shape[1] / 2),
            int(fragment_match_point[1] - fragment.shape[0] / 2)
        )

        frag_height = fragment.shape[0]
        frag_width = fragment.shape[1]

        # The new position is the fresco match point + the fragment match point
        frag_position_x = fresco_match_point[0] + fragment_match_point[0]
        frag_position_y = fresco_match_point[1] + fragment_match_point[1]

        fragment_result = cv.circle(
            fragment_result, fragment_match_point, 5, (255, 0, 0), 2)
        add_fragment_to_target_image({"x": frag_position_x, "y": frag_position_y},
                                     fragment_result, background, background.shape[1], background.shape[0])

        background = cv.circle(
            background, fresco_match_point, 5, (0, 0, 255), 2)

    cv.imshow(fragment_path, background)
    cv.waitKey()
    cv.destroyAllWindows()

    print(ok_frag_kp_nb, "/", len(fragments),
          math.floor((ok_frag_kp_nb / len(fragments)) * 100), "% kp ok")
    print(ok_frag_match_nb, "/", ok_frag_kp_nb,
          math.floor((ok_frag_match_nb / ok_frag_kp_nb) * 100), "% match ok")
