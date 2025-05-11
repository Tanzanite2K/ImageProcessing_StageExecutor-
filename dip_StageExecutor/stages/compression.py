
import cv2

def jpeg_compression(img_path, quality=30):
    # Lossy JPEG compression
    cv2.imwrite("compressed_lossy.jpg", cv2.imread(img_path), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imread("compressed_lossy.jpg")

def png_compression(img_path):
    # Lossless PNG compression
    cv2.imwrite("compressed_lossless.png", cv2.imread(img_path), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    return cv2.imread("compressed_lossless.png")
def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()
def run():
    # Load the original image
    img_path = "input_images\\lennaimage.jpeg"
    
    # Perform JPEG compression (lossy)
    compressed_lossy = jpeg_compression(img_path, 30)
    show("Compressed Lossy (JPEG)", compressed_lossy)
   

    # Perform PNG compression (lossless)
    compressed_lossless = png_compression(img_path)
    show("Compressed Lossless (PNG)", compressed_lossless)

    cv2.destroyAllWindows()

