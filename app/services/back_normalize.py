import cv2
import numpy as np

def align_image_with_references(image, reference_images, nfeatures=5000, min_matches=20, ransac_threshold=5.0, lowe_ratio=0.7):
    """
    Aligns the given image (as a numpy array) with cached reference images using SIFT.
    """
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)

    keypoints_original, descriptors_original = sift.detectAndCompute(gray_original, None)
    best_aligned_image = None
    max_inliers = 0

    for image_reference in reference_images:
        gray_reference = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY)
        keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_reference, None)

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(descriptors_original, descriptors_ref, k=2)

        good_matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]

        if len(good_matches) > min_matches:
            src_pts = np.float32([keypoints_original[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
            inliers = np.sum(mask) if mask is not None else 0

            if inliers > max_inliers:
                max_inliers = inliers
                best_aligned_image = cv2.warpPerspective(image, M, (image_reference.shape[1], image_reference.shape[0]))

    return best_aligned_image
