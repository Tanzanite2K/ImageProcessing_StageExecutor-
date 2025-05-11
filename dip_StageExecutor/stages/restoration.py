
import cv2
import numpy as np

# Add Salt and Pepper noise to the image
def add_salt_pepper_noise(img, amount=0.05):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255

    num_pepper = np.ceil(amount * img.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

# Mean filter
def mean_filter(img):
    return cv2.blur(img, (5, 5))

# Median filter
def median_filter(img):
    return cv2.medianBlur(img, 5)

# Adaptive median filter
def adaptive_median_filter(img):
    return cv2.medianBlur(img, 7)

# Weighted Average filter (using a simple kernel)
def weighted_average_filter(img):
    kernel = np.ones((5, 5), np.float32) / 25  # Simple averaging kernel
    return cv2.filter2D(img, -1, kernel)

# Min filter (using a minimum filter approach)
def min_filter(img):
    return cv2.erode(img, np.ones((5, 5), np.uint8))

# Max filter (using a maximum filter approach)
def max_filter(img):
    return cv2.dilate(img, np.ones((5, 5), np.uint8))

# Gaussian filter (with a 5x5 kernel and sigma value)
def gaussian_filter(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

# Show image function
def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to run all filters and show results
def run():
    img = cv2.imread("input_images\\lennaimage.jpeg", cv2.IMREAD_GRAYSCALE)
    noisy = add_salt_pepper_noise(img)
    
    show("Original", img)
    show("Noisy", noisy)
    show("Mean Filter", mean_filter(noisy))
    show("Median Filter", median_filter(noisy))
    show("Adaptive Median Filter", adaptive_median_filter(noisy))
    show("Weighted Average Filter", weighted_average_filter(noisy))
    show("Min Filter", min_filter(noisy))
    show("Max Filter", max_filter(noisy))
    show("Gaussian Filter", gaussian_filter(noisy))

    cv2.destroyAllWindows()

