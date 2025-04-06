from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def stain_standardization(img):

    img = img.astype(np.float64)
    mean = np.mean(img, axis=(0, 1), keepdims=True)  # Per-channel mean
    std = np.std(img, axis=(0, 1), keepdims=True)  # Per-channel std

    # Avoid division by zero
    std = np.where(std == 0, 1e-6, std)

    result = (img - mean) / std
    #print(result)
    return result  # * 255.0
