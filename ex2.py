"""Exercise 2
RANSAC filter to place the fragments on the right coordinates and rotation

Authors: Tom Mansion <tom.mansion@universite-paris-saclay.fr>, Sophie Nguyen <sophie.nguyen@universite-paris-saclay.fr>
"""

import math
import random as rd
import cv2 as cv
import numpy as np
import csv

from utils import MICHELANGELO_IMG, MICHELANGELO_PATH, rotate_fragment, get_fragment_number, get_image_background, add_fragment_to_target_image, get_ORB_interest_points, get_all_fragments_path, get_BFMatcher_associations

FRESCO_IMG = MICHELANGELO_IMG
FRESCO_PATH = MICHELANGELO_PATH

GOOD_MATCH_PERCENT = 0.3
MATCHER_THRESHOLD = 0.9
THRESHOLD_RADIUS = 15
MAX_ITER = 10000
MATCHES_N = 2

class Model:
    """Possible fragment model
    """
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        return f"{self.x} {self.y} {math.degrees(self.theta)}"


def get_relevant_matches(frag_img, matches, img_keypoints, frag_keypoints, n=MATCHES_N, max_i=MAX_ITER, threshold_radius=THRESHOLD_RADIUS):
    """Apply RANSAC filter to matches to compute a fragment model
    Pick the model with the largest number of inliers.

    Args:
        frag_img (np.array): Fragment image
        matches (list): List of key points matches between the fragment and the fresco
        img_keypoints (list): List of detected key points on the fresco
        frag_keypoints (list): List of detected key points on the fragment
        n (int, optional): Number of random matches required to compute a model. Defaults to MATCHES_N.
        max_i (int, optional): Maximum number of iterations (draws). Defaults to MAX_ITER.
        threshold_radius (int, optional): Circle radius around a key point to define if coordinates match with it. Defaults to THRESHOLD_RADIUS.

    Returns:
        list, model: Inliers matches (without the drawn matches) and the best model
    """
    cols_f, rows_f = frag_img.shape[:2]
    if len(matches) == 0:
        return
    models = []
    matches_sets = []

    for i in range(max_i):
        random_matches = []
        remaining_matches = matches.copy()
        while len(random_matches) < n:
            index = rd.randint(0, len(remaining_matches) - 1)
            if remaining_matches[index] in random_matches:
                continue
            random_matches.append(matches[index][0])
            del remaining_matches[index]

        model, t = get_model(
            random_matches[0], random_matches[1], img_keypoints, frag_keypoints, cols_f, rows_f)

        if (len(remaining_matches) == 0):
            return [], model
        
        models.append(model)

        model_matches = []
        for match in remaining_matches:
            kp_img, kp_frag = get_match_keypoints_coords(
                match[0], img_keypoints, frag_keypoints)
            kp_check_img = get_translated_coordinates(kp_frag, t, model.theta)
            if is_in_circle(kp_check_img, kp_img, threshold_radius):
                model_matches.append(match[0])
        matches_sets.append(model_matches)
    best_model_index = np.array(
        list(map(lambda m_set: len(m_set), matches_sets))).argmax()
    best_model = models[best_model_index]
    inliers_matches = matches_sets[best_model_index]
    return inliers_matches, best_model

def is_in_circle(pt, center, radius):
    """Determine if a point is in a circle with a defined center and radius

    Args:
        pt (tuple): Point coordinates
        center (tuple): Circle center coordinates
        radius (int): Circle radius

    Returns:
        boolean: Answer
    """
    return (pt[0] - center[0]) ** 2 + (pt[1] - center[1]) ** 2 < radius ** 2

def get_match_keypoints_coords(match, source_keypoints, frag_keypoints):
    """Retrieve the coordinates of a match (DMatch class)

    Args:
        match (DMatch): Association of two key points
        source_keypoints (list): List of key points on the fresco
        frag_keypoints (list): List of key points on the fragment

    Returns:
        tuple, tuple: Coordinates of key points
    """
    return source_keypoints[match.trainIdx].pt, frag_keypoints[match.queryIdx].pt

def get_intersection_angle(A_frag, B_frag, A_img, B_img):
    """Compute intersection angle between the line AB on the fresco and the line A'B' on the

    Args:
        A_frag (tuple): Coordinates of point A' on the fragment
        B_frag (tuple): Coordinates of point B' on the fragment
        A_img (tuple): Coordinates of point A on the fresco
        B_img (tuple): Coordinates of point B on the fresco

    Returns:
        double: Angle theta
    """
    u = (B_frag[0] - A_frag[0], B_frag[1] - A_frag[1])
    v = (B_img[0] - A_img[0], B_img[1] - A_img[1])

    if (u == v):
        return 0

    norm_u = math.sqrt((u[0] ** 2) + (u[1] ** 2))
    norm_v = math.sqrt((v[0] ** 2) + (v[1] ** 2))

    norm_product = norm_u * norm_v

    if norm_product == 0:
        return 0

    return math.acos(round((u[0] * v[0] + u[1] * v[1]) / (norm_product), 2))

def get_translation(pt_img, pt_origin):
    """Compute translation t (difference between real and origin coordinates)

    Args:
        pt_img (tuple): Real coordinates of a point
        pt_origin (tuple): Coordinates of a point if the fragment is on the origin

    Returns:
        np.array: Translation t
    """
    return np.array([pt_img[0] - pt_origin[0], pt_img[1] - pt_origin[1]])

def get_translated_coordinates(pt, t, theta):
    """Get the real coordinates of a fragment point on the fresco

    Args:
        pt (tuple): Fragment point coordinates
        t (np.array): Translation to apply
        theta (double): Rotation angle to apply

    Returns:
        tuple: Real coordinates
    """
    rotated_pt = np.matmul(get_rotation_matrix(theta), np.array(pt))
    return (t[0] + rotated_pt[0], t[1] + rotated_pt[1])

def get_rotation_matrix(theta):
    """Get rotation matrix in a left-handed coordinate system

    Args:
        theta (double): Angle in radian

    Returns:
        np.array: Rotation matrix associated with the angle
    """
    return np.array(
        (
            (np.cos(theta), np.sin(theta)),
            (-np.sin(theta),  np.cos(theta)))
    )

def get_model(A_match, B_match, img_keypoints, frag_keypoints, cols_f, rows_f):
    """Compute a fragment model according to the parameters

    Args:
        A_match (DMatch): First point association
        B_match (DMatch): Second point association
        img_keypoints (list): List of key points on the fresco
        frag_keypoints (list): List of key points on the fragment
        cols_f (int): Fragment width
        rows_f (int): Fragment height

    Returns:
        model, t: Model and translation to apply to each point
    """
    A_img, A_frag = get_match_keypoints_coords(
        A_match, img_keypoints, frag_keypoints)
    B_img, B_frag = get_match_keypoints_coords(
        B_match, img_keypoints, frag_keypoints)
    theta = get_intersection_angle(A_frag, B_frag, A_img, B_img)
    center_frag = ((cols_f - 1) / 2.0, (rows_f - 1) / 2.0)
    r = get_rotation_matrix(theta)
    A_frag_rotated = np.matmul(r, np.array(A_frag))
    t = get_translation(A_img, A_frag_rotated)
    center_img = get_translated_coordinates(center_frag, t, theta)

    return Model(int(center_img[0]), int(center_img[1]), theta), t

def get_model_regression(frag_img, initial_model, inliers_matches, img_keypoints, frag_keypoints):
    cols_f, rows_f = frag_img.shape[:2]
    models = [initial_model]
    for i in range(len(inliers_matches)):
        model_matches = [inliers_matches[i]]
        if i + 1 != len(inliers_matches):
            for j in range(i + 1, len(inliers_matches)):
                model_matches.append(inliers_matches[j])
                model, _ = get_model(
                    model_matches[0], model_matches[1], img_keypoints, frag_keypoints, cols_f, rows_f)
                models.append(model)
    x_avg = sum(model.x for model in models) / float(len(models))
    y_avg = sum(model.y for model in models) / float(len(models))
    theta_avg = sum(model.theta for model in models) / float(len(models))
    return Model(int(x_avg), int(y_avg), theta_avg)

def get_good_matches(matches, good_percent):
    """Retrieve only an amount of best matches (according to DMatch.distance)

    Args:
        matches (list): List of associations
        good_percent (double): Percentage to keep

    Returns:
        list: List of best matches
    """
    matches.sort(key=lambda m: m[0].distance)
    good_matches_count = int(len(matches) * good_percent)
    return matches[0:good_matches_count]

def main():
    # Get the fresco
    fresco = cv.imread(
        FRESCO_IMG, cv.IMREAD_GRAYSCALE)

    background = get_image_background(FRESCO_IMG)

    # Get the ORB interest points
    fresco_kp, fresco_des = get_ORB_interest_points(
        fresco, number_of_keypoints=30000)

    print("fresco_kp: ", len(fresco_kp))
    # Get the fragments
    fragments = get_all_fragments_path(FRESCO_PATH)
    print("Number of fragments : ", len(fragments))

    ok_frag_kp_nb = 0
    ok_frag_match_nb = 0
    solution_export = []  # For the ex3 [frag_num, x, y, theta]

    for fragment_path in fragments:  # :10]:
        print("  ", fragment_path)

        # Load the fragment image
        fragment = cv.imread(fragment_path, cv.IMREAD_GRAYSCALE)

        # Get the ORB interest points
        fragment_kp, fragment_des = get_ORB_interest_points(
            fragment, number_of_keypoints=30000)
        print("   fragment_kp: ", len(fragment_kp))

        if len(fragment_kp) < 2:
            print("        ! Not enough keypoints")
            continue

        ok_frag_kp_nb += 1

        # Get the BFMatcher associations
        matches = get_BFMatcher_associations(
            fragment_des, fresco_des, MATCHER_THRESHOLD)

        matches = get_good_matches(matches, GOOD_MATCH_PERCENT)

        print("        matches: ", len(matches))
        if len(matches) < MATCHES_N:
            print("             ! Not enough matches.")
            continue

        ok_frag_match_nb += 1

        inliers_matches, best_model = get_relevant_matches(
            fragment, matches, fresco_kp, fragment_kp)

        # Question 2.3
        if len(inliers_matches) > MATCHES_N:
            best_model = get_model_regression(
                fragment, best_model, inliers_matches, fresco_kp, fragment_kp)

        print(f"        inliers matches: {len(inliers_matches)}")
        print(f"        model: {best_model}")

        unchanged_fragment_img = cv.imread(fragment_path, cv.IMREAD_UNCHANGED)
        cols_f, rows_f = unchanged_fragment_img.shape[:2]
        rotated_fragment = rotate_fragment(
            unchanged_fragment_img, math.degrees(best_model.theta), cols_f, rows_f)

        add_fragment_to_target_image({"x": best_model.x, "y": best_model.y},
                                     rotated_fragment, background, background.shape[1], background.shape[0])

        # Enable parameter for fragment number writing (debugging)
        # cv.putText(
        #     background,
        #     get_fragment_number(fragment_path),
        #     (best_model.x, best_model.y),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 255, 255),
        #     2
        # )

        # Question 3
        solution_export.append(
            [
                get_fragment_number(fragment_path),
                best_model.x,
                best_model.y,
                math.degrees(best_model.theta) - 360
            ])

    cv.imshow("Fresco with the fragments", background)
    cv.waitKey()
    cv.destroyAllWindows()

    print(ok_frag_kp_nb, "/", len(fragments),
          math.floor((ok_frag_kp_nb / len(fragments)) * 100), "% kp ok")
    print(ok_frag_match_nb, "/", ok_frag_kp_nb,
          math.floor((ok_frag_match_nb / ok_frag_kp_nb) * 100), "% match ok")

    # Question 3 - Export solution in txt (csv with space delimiter)
    csv_file_name = "solution_ORB_BFMATCHER_" + str(MAX_ITER) + "_MAX_ITER.txt"
    with open(csv_file_name, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
        writer.writerows(solution_export)

if __name__ == "__main__":
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
    stats.dump_stats('./stats.prof')
