import os
import shutil
import numpy as np
from tensorflow.keras.models import model_from_json
from pcam_loader.pcam_loader import PCAMDataLoader
from config import GLOBAL_DATASET_PATH

class FalsePositiveSaver:
    def __init__(self, base_dir, json_model_path, weights_path, output_dir,
                 image_size=96, batch_size=1):
        self.base_dir = base_dir
        self.json_model_path = json_model_path
        self.weights_path = weights_path
        self.output_dir = output_dir
        self.image_size = image_size
        self.batch_size = batch_size
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = self._load_model()
        self.loader = PCAMDataLoader(base_dir=self.base_dir, image_size=self.image_size, standardize=False)
        _, self.val_gen = self.loader.get_generators(val_batch_size=self.batch_size, train_batch_size=self.batch_size, class_mode='binary')

    def _load_model(self):
        with open(self.json_model_path, "r") as f:
            model = model_from_json(f.read())
        model.load_weights(self.weights_path)
        return model

    @staticmethod
    def stain_standardization(img):
        img = img.astype(np.float64)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        std = np.std(img, axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1e-6, std)
        return (img - mean) / std

    def run(self):
        total = 0
        saved = 0
        filenames = self.val_gen.filenames
        valid_dir = self.val_gen.directory

        for i in range(len(self.val_gen)):
            images, labels = self.val_gen[i]
            standardized_images = self.stain_standardization(images)
            predictions = self.model.predict(standardized_images, verbose=0)

            for j in range(len(labels)):
                true_label = int(labels[j])
                pred_prob = predictions[j][0]
                pred_label = 1 if pred_prob >= 0.5 else 0

                if true_label == 0 and pred_label == 1:
                    confidence = int(100 - (pred_prob * 100))
                    relative_path = filenames[total]
                    original_filename = os.path.basename(relative_path)
                    original_file_path = os.path.join(valid_dir, relative_path)

                    new_filename = f"{confidence:03d}_{original_filename}"
                    dest_path = os.path.join(self.output_dir, new_filename)
                    shutil.copy2(original_file_path, dest_path)
                    saved += 1

                total += 1

        print(f"Processed {total} samples. Saved {saved} false positives to '{self.output_dir}'.")



saver = FalsePositiveSaver(
    base_dir=GLOBAL_DATASET_PATH,
    json_model_path="../models/gradcam_stain_standardization.json",
    weights_path="../models/gradcam_stain_standardization.hdf5",
    output_dir="../false_positives"
)
saver.run()
