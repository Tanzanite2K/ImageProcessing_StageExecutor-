

import cv2
import numpy as np

def negative_image(img):
    return 255 - img

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img.astype(np.float32))
    return np.uint8(log_img)

def gamma_correction(img, gamma=0.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def contrast_stretch(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def bit_plane_slicing(img):
    rows, cols = img.shape
    planes = []
    for i in range(8):  # Only extracting the first 3 bit planes
        plane = (img & (1 << i)) >> i
        planes.append(plane * 255)
    return planes

def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run():
    img = cv2.imread(r"input_images\lennaimage.jpeg", cv2.IMREAD_GRAYSCALE)

    show("Original", img)
    show("Negative", negative_image(img))
    show("Log Transform", log_transform(img))
    show("Gamma Correction", gamma_correction(img, 0.4))
    show("Contrast Stretch", contrast_stretch(img))
    show("Histogram Equalization", histogram_equalization(img))

    planes = bit_plane_slicing(img)
    for i, plane in enumerate(planes):
        show(f"Bit Plane {i}", plane)
