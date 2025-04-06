import base64
import io
import matplotlib.pyplot as plt
import os
from config import FALSE_POSITIVES_PATH
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess_image(image_path, target_size=(96, 96)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    return img_array

def img_to_base64(img_array, size=(96, 96), title=""):
    fig, ax = plt.subplots(figsize=(size[0] / 32, size[1] / 32))
    ax.imshow(img_array)
    ax.axis("off")
    buf = io.BytesIO()
    plt.title(title, color="white")
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#343a40")
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

def load_images():
    if not os.path.exists(FALSE_POSITIVES_PATH):
        raise FileNotFoundError(f"⚠ Folder '{FALSE_POSITIVES_PATH}' does not exist! Run the filtering script first.")

    image_list = []
    for filename in sorted(os.listdir(FALSE_POSITIVES_PATH)):  # Sorted by confidence
        img_path = os.path.join(FALSE_POSITIVES_PATH, filename)
        img_array = preprocess_image(img_path)
        image_list.append((img_array, filename))

    if not image_list:
        raise ValueError("⚠ No images found in 'false_positives/'. Run the filtering script first!")

    return image_list