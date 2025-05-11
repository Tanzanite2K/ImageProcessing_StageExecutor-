

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

def read_image():
    print("Reading image...")
    img = cv2.imread(r"input_images\lennaimage.jpeg")  # raw string to avoid warning
    return img

def display_image(img, title="Image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def intensity_scaling(img, scale=1.2):
    print(f"Applying intensity scaling with scale: {scale}")
    return np.clip(img * scale, 0, 255).astype(np.uint8)

def spatial_scaling(img, scale=1.0):
    print(f"Applying spatial scaling with scale: {scale}")
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height))

def read_image_metadata(image_path):
    print("Reading image metadata...")
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            print("No metadata found.")
            return
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            print(f"{tag}: {value}")
    except Exception as e:
        print(f"Failed to read metadata: {e}")

def resize_image(img, scale=0.5):
    print(f"Final resizing of image by scale {scale}")
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height))

def run():
    image_path = r"input_images\lennaimage.jpeg"
    img = read_image()
    display_image(img, "Original")

    read_image_metadata(image_path)

    intensity_img = intensity_scaling(img, scale=1.1)
    display_image(intensity_img, "Intensity Scaled")

    spatial_img = spatial_scaling(intensity_img, scale=0.8)
    display_image(spatial_img, "Spatially Scaled")

    resized_img = resize_image(spatial_img, scale=0.5)
    display_image(resized_img, "Final Resized")
