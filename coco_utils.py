from pycocotools.mask import encode
import numpy as np
import cv2


def encode_mask(mask: np.ndarray):
    fortran_binary_mask = np.asfortranarray(mask)
    encoded_mask = encode(fortran_binary_mask)
    return encoded_mask


def bounding_box_from_mask(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h
