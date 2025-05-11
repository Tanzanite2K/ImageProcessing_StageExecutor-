

import cv2
import numpy as np

def edge_detection_sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    return cv2.convertScaleAbs(sobelx + sobely)

def edge_detection_prewitt(img):
    kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    grad_x = cv2.filter2D(img, cv2.CV_64F, kernelx)
    grad_y = cv2.filter2D(img, cv2.CV_64F, kernely)
    
    return cv2.convertScaleAbs(grad_x + grad_y)

def edge_detection_roberts(img):
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    
    grad_x = cv2.filter2D(img, cv2.CV_64F, kernelx)
    grad_y = cv2.filter2D(img, cv2.CV_64F, kernely)
    
    return cv2.convertScaleAbs(grad_x + grad_y)

def canny_edge(img):
    return cv2.Canny(img, 100, 200)

def otsu_threshold(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

def run():
    img = cv2.imread("input_images/lennaimage.jpeg", cv2.IMREAD_GRAYSCALE)

    show("Original", img)
    show("Sobel Edges", edge_detection_sobel(img))
    show("Prewitt Edges", edge_detection_prewitt(img))
    show("Roberts Edges", edge_detection_roberts(img))
    show("Canny Edges", canny_edge(img))
    show("Otsu Threshold", otsu_threshold(img))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


