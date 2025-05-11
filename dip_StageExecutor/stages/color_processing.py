
import cv2
import numpy as np

def convert_to_cmy(img):
    bgr_norm = img / 255.0
    cmy = 1 - bgr_norm
    return (cmy * 255).astype(np.uint8)

def convert_to_cmyk(img):
    bgr_norm = img / 255.0
    k = 1 - np.max(bgr_norm, axis=2)
    k_safe = k + 1e-5  # Avoid division by zero
    c = (1 - bgr_norm[..., 2] - k) / k_safe
    m = (1 - bgr_norm[..., 1] - k) / k_safe
    y = (1 - bgr_norm[..., 0] - k) / k_safe

    cmyk = np.stack([c, m, y, k], axis=2)
    return (cmyk * 255).astype(np.uint8)

def convert_to_hsi(img):
    img = img.astype(np.float32) / 255.0
    B, G, R = cv2.split(img)

    # Intensity
    I = (R + G + B) / 3

    # Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 * min_rgb / (R + G + B + 1e-6))

    # Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1, 1))

    H = np.zeros_like(R)
    H[B <= G] = theta[B <= G]
    H[B > G] = (2 * np.pi - theta[B > G])
    H = H / (2 * np.pi)  # Normalize to [0,1]

    HSI = cv2.merge((H, S, I))
    return (HSI * 255).astype(np.uint8)

def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

def run():
    img = cv2.imread("input_images\\lennaimage.jpeg")
    if img is None:
        print("Error: Image not found!")
        return

    cmy = convert_to_cmy(img)
    cmyk = convert_to_cmyk(img)
    hsi = convert_to_hsi(img)

    # Display the converted images
    show("CMY Image", cmy)
    show("CMYK Image", cmyk)
    
    # Split HSI channels for better visualization
    h, s, i = cv2.split(hsi)
    show("HSI - Hue", h)
    show("HSI - Saturation", s)
    show("HSI - Intensity", i)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

