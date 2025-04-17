import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

IMAGE_SIZE = 96

class PCAMDataLoader:
    def __init__(self, base_dir, image_size=IMAGE_SIZE, standardize=False):
        self.base_dir = base_dir
        self.image_size = image_size
        if standardize:
            self.datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=self.stain_standardization)
        else:
            self.datagen = ImageDataGenerator(rescale=1./255)

    def stain_standardization(self, img):
        img = img.astype(np.float64)
        mean = np.mean(img, axis=(0, 1), keepdims=True)  # Per-channel mean
        std = np.std(img, axis=(0, 1), keepdims=True)  # Per-channel std

        # Avoid division by zero
        std = np.where(std == 0, 1e-6, std)

        result = (img - mean) / std
        # print(result)
        return result * 255.0

    def get_generators(self, train_batch_size=32, val_batch_size=32, shuffle=False, class_mode='binary', train_val=True):
        if train_val:
            train_path = os.path.join(self.base_dir, 'train+val', 'train')
            valid_path = os.path.join(self.base_dir, 'train+val', 'valid')
        else:
            train_path = os.path.join(self.base_dir)
            valid_path = os.path.join(self.base_dir)

        train_gen = self.datagen.flow_from_directory(
            train_path,
            target_size=(self.image_size, self.image_size),
            batch_size=train_batch_size,
            class_mode=class_mode,
            shuffle=shuffle
        )

        val_gen = self.datagen.flow_from_directory(
            valid_path,
            target_size=(self.image_size, self.image_size),
            batch_size=val_batch_size,
            class_mode=class_mode,
            shuffle=shuffle
        )

        return train_gen, val_gen
