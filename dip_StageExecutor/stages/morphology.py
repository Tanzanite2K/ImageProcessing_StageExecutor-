
import cv2
import numpy as np

# Thresholding function
def threshold(img):
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary

# Erosion function
def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

# Dilation function
def dilation(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

# Opening (erosion followed by dilation) function
def opening(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Closing (dilation followed by erosion) function
def closing(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Show image function with a wait before showing the next one
def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

# Main function to run all transformations and show results
def run():
    img = cv2.imread("input_images\\lennaimage.jpeg", cv2.IMREAD_GRAYSCALE)
    binary = threshold(img)
    
    # Display each image one by one
    show("Binary", binary)
    show("Erosion", erosion(binary))
    show("Dilation", dilation(binary))
    show("Opening", opening(binary))
    show("Closing", closing(binary))

    cv2.destroyAllWindows()
