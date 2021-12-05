"""Exercise 1 (Pt. 1)
For each fragment, compute associations with the fresco using KNN matcher and show the matches.

Authors: Tom Mansion <tom.mansion@universite-paris-saclay.fr>, Sophie Nguyen <sophie.nguyen@universite-paris-saclay.fr>
"""

import os
import math
import cv2 as cv

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

    # Find the keypoints and descriptors with ORB
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


# Main : test the methods
if __name__ == "__main__":

    # Get the fresco
    fresco = cv.imread(
        "Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg", cv.IMREAD_GRAYSCALE)

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

        # Draw the matches
        img3 = cv.drawMatchesKnn(fragment, fragment_kp, fresco, fresco_kp,
                                 matches[:2], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # if the 'q' key is pressed, stop the loop
        cv.imshow(fragment_path, img3)
        if cv.waitKey() & 0xFF == ord("q"):
            break
        cv.destroyAllWindows()

    print(ok_frag_kp_nb, "/", len(fragments),
          math.floor((ok_frag_kp_nb / len(fragments)) * 100), "% kp ok")
    print(ok_frag_match_nb, "/", ok_frag_kp_nb,
          math.floor((ok_frag_match_nb / ok_frag_kp_nb) * 100), "% match ok")
