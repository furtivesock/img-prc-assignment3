import math
import random as rd
import cv2 as cv
import matplotlib.pyplot as plt
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
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        return f"{self.x} {self.y} {math.degrees(self.theta)}"


def get_relevant_matches(frag_img, matches, img_keypoints, frag_keypoints, n=MATCHES_N, max_i=MAX_ITER, threshold_radius=THRESHOLD_RADIUS):
    # TODO: Initialize
    cols_f, rows_f = frag_img.shape[:2]
    if len(matches) == 0:
        return
    models = []
    matches_sets = []

    # TODO: Iterator to max_i
    for i in range(max_i):
        random_matches = []
        random_indexes = []
        # TODO: Pick n matches randomly (sampling)
        remaining_matches = matches.copy()
        # TODO: Fix problem
        if len(matches) == n:
            random_matches.append(matches[0][0])
            random_matches.append(matches[1][0])
        else:
            while len(random_indexes) < n:
                index = rd.randint(0, len(remaining_matches) - 1)
                if index in random_indexes:
                    continue
                random_indexes.append(index)
                random_matches.append(matches[index][0])
                del remaining_matches[index]

        model, t = get_model(
            random_matches[0], random_matches[1], img_keypoints, frag_keypoints, cols_f, rows_f)
        models.append(model)

        model_matches = []
        for match in remaining_matches:
            kp_img, kp_frag = get_match_keypoints_coords(
                match[0], img_keypoints, frag_keypoints)
            kp_check_img = get_translated_coordinates(kp_frag, t, model.theta)
            if is_in_circle(kp_check_img, kp_img, threshold_radius):
                model_matches.append(match[0])
        matches_sets.append(model_matches)
    # TODO: Pick the model which has the best score
    # TODO: Voting with the pixels colors
    best_model_index = np.array(
        list(map(lambda m_set: len(m_set), matches_sets))).argmax()
    best_model = models[best_model_index]
    inliers_matches = matches_sets[best_model_index]
    return inliers_matches, best_model


def is_in_circle(pt, center, radius):
    return (pt[0] - center[0]) ** 2 + (pt[1] - center[1]) ** 2 < radius ** 2


def get_match_keypoints_coords(match, source_keypoints, frag_keypoints):
    return source_keypoints[match.trainIdx].pt, frag_keypoints[match.queryIdx].pt


def get_intersection_angle(A_frag, B_frag, A_img, B_img):
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
    return np.array([pt_img[0] - pt_origin[0], pt_img[1] - pt_origin[1]])


def get_translated_coordinates(pt, t, theta):
    rotated_pt = np.matmul(get_rotation_matrix(theta), np.array(pt))
    return (t[0] + rotated_pt[0], t[1] + rotated_pt[1])


def get_rotation_matrix(theta):
    """Get rotation matrix in a left-handed coordinate system

    Args:
        theta (double): Angle in radian

    Returns:
        [type]: [description]
    """
    return np.array(
        (
            (np.cos(theta), np.sin(theta)),
            (-np.sin(theta),  np.cos(theta)))
    )


def get_model(A_match, B_match, img_keypoints, frag_keypoints, cols_f, rows_f):
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
        if i + 1 != len(inliers_matches) - 1:
            for j in range(i + 1, len(inliers_matches)):
                model_matches.append(inliers_matches[j])
                model, _ = get_model(
                    model_matches[0], model_matches[1], img_keypoints, frag_keypoints, cols_f, rows_f)
                models.append(model)
    x_avg = sum(model.x for model in models) / float(len(models))
    y_avg = sum(model.y for model in models) / float(len(models))
    theta_avg = sum(model.theta for model in models) / float(len(models))
    return Model(int(x_avg), int(y_avg), theta_avg)


def is_similar_color(frag_pt_area, img_pt_area, threshold):
    pass


def get_good_matches(matches, good_percent):
    # TODO: Sort by best distance (similarity) (the lower, the best)
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
        if len(matches) < 2:
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

        # TODO: Enable parameter for fragment number writing

        cv.putText(
            background,
            get_fragment_number(fragment_path),
            (best_model.x, best_model.y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255, 255),
            2
        )

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
