

import cv2
import numpy as np
from skimage.measure import label, regionprops

def compute_area(cnt):
    return cv2.contourArea(cnt)

def compute_bounding_rect(cnt):
    return cv2.boundingRect(cnt)  # x, y, w, h

def compute_rectangularity(area, bounding_rect):
    _, _, w, h = bounding_rect
    rect_area = w * h
    return area / rect_area if rect_area != 0 else 0

def compute_compactness(area, bounding_rect):
    _, _, w, h = bounding_rect
    perimeter_approx = 2 * (w + h)
    return (4 * np.pi * area) / (perimeter_approx ** 2) if perimeter_approx != 0 else 0

def convex_hull_display(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_img = np.zeros_like(img)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(hull_img, [hull], -1, 255, 2)
    return hull_img

def analyze_shapes(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labeled_img = label(img)
    props = regionprops(labeled_img)

    print("\n--- Shape Features ---")
    for i, cnt in enumerate(contours):
        area = compute_area(cnt)
        if area < 100:
            continue

        bounding_rect = compute_bounding_rect(cnt)
        rectangularity = compute_rectangularity(area, bounding_rect)
        compactness = compute_compactness(area, bounding_rect)

        if i >= len(props):
            continue

        region = props[i]
        eccentricity = region.eccentricity
        elongatedness = (
            region.major_axis_length / region.minor_axis_length
            if region.minor_axis_length != 0 else 0
        )
        euler_number = region.euler_number

        print(f"\nShape {i + 1}")
        print(f"Area: {area}")
        print(f"Eccentricity: {eccentricity:.4f}")
        print(f"Euler Number: {euler_number}")
        print(f"Elongatedness: {elongatedness:.4f}")
        print(f"Compactness: {compactness:.4f}")
        print(f"Rectangularity: {rectangularity:.4f}")

def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

def run():
    img = cv2.imread("input_images/lennaimage.jpeg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found.")
        return

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    analyze_shapes(binary)
    hull_img = convex_hull_display(binary)

    show("Original", img)
    show("Convex Hull", hull_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
