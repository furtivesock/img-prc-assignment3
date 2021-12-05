"""Common functions used by the other files

Authors: Tom Mansion <tom.mansion@universite-paris-saclay.fr>, Sophie Nguyen <sophie.nguyen@universite-paris-saclay.fr>
"""

import os
import numpy as np
import cv2 as cv
import re

MICHELANGELO_IMG = "Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg"
DOMENICHO_IMG = "Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn.jpg"

MICHELANGELO_PATH = "Michelangelo"
DOMENICHO_PATH = "Domenichino_Virgin-and-unicorn"

ORIGINAL_IMAGE_OPACITY = 0.3


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


def get_image_background(image_path):
    """Set a light background with the original image
    It will help us understand how the fragments match
    Add a low alpha channel to the original image

    Args:
        image_path (string): Path of the image

    Returns:
        np.array: Whiter image
    """
    image = cv.imread(
        image_path, cv.IMREAD_UNCHANGED)
    target_height, target_width, _ = image.shape
    white_background = np.full(
        (target_height, target_width, 3), 255, dtype=np.uint8)
    return cv.addWeighted(
        image, ORIGINAL_IMAGE_OPACITY, white_background, 1 - ORIGINAL_IMAGE_OPACITY, 0)


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


def get_fragment_number(fragment_path):
    p = re.compile('(\d+)\.png')
    m = p.search(fragment_path)
    return m.group(1)


def rotate_fragment(fragment_img, angle, cols_f, rows_f) -> np.array:
    """Rotate a fragment by the angle

    Args:
        fragment_img (2D array):    Image of a fragment
        angle (double):             Angle (in radians) to rotate
        cols_f (int):               Fragment width
        rows_f (int):               Fragment height

    Returns:
        2D array: Rotated fragment
    """
    # Calculate the coordinates of the center of the fragment
    certer_coordinates = ((cols_f - 1) / 2.0, (rows_f - 1) / 2.0)
    # Create a rotation matrix
    M_rotation = cv.getRotationMatrix2D(certer_coordinates, angle, 1)
    # Apply the rotation matrix on the fragment
    rotated_fragment = cv.warpAffine(
        fragment_img, M_rotation, (cols_f, rows_f))
    return rotated_fragment
