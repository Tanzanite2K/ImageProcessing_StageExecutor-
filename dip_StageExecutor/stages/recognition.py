import cv2
import numpy as np

def connected_components(img):
    num_labels, labels_im = cv2.connectedComponents(img)
    print("Number of objects:", num_labels - 1)
    return labels_im

def nearest_neighbor_classifier(features, sample):
    distances = [np.linalg.norm(np.array(f) - np.array(sample)) for f in features]
    return np.argmin(distances)

def run():
    img = cv2.imread("input_images\lennaimage.jpeg", cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    labels = connected_components(binary)
    labeled_display = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Dummy classification example
    sample = [500, 300]  # area, perimeter
    features = [[480, 290], [510, 310]]
    result = nearest_neighbor_classifier(features, sample)
    print("Classified as class index:", result)

    cv2.imshow("Labeled Image", labeled_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
