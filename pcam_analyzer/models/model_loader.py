import os
import cv2
import numpy as np
import torchstain
from torchvision import transforms
from tensorflow.keras.models import model_from_json
from config import DEFAULT_MODEL_JSON, DEFAULT_MODEL_WEIGHTS

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])


def load_model(json_path=DEFAULT_MODEL_JSON, weights_path=DEFAULT_MODEL_WEIGHTS):
    with open(json_path, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)
    return model
