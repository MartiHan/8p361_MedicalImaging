import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.image import save_img
from pcam_loader.pcam_loader import PCAMDataLoader
from config import GLOBAL_DATASET_PATH

class FewShotSubsetSelector:
    def __init__(self, base_dir, image_size=96, batch_size=32):
        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_gen, self.val_gen = self._load_data()

    def _load_data(self):
        loader = PCAMDataLoader(base_dir=self.base_dir, image_size=self.image_size)
        return loader.get_generators(train_batch_size=self.batch_size, val_batch_size=self.batch_size)

    def encoder_blocks(self, inputs):
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='gradcam_layer')(x)
        x = layers.BatchNormalization(name='bn4')(x)

        return x

    def get_frozen_encoder(self, input_shape=(96, 96, 3), weights_path="encoder_classifier_weights.h5"):
        inputs = Input(shape=input_shape)
        x = self.encoder_blocks(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        model = Model(inputs, x, name="frozen_encoder")
        model.load_weights(weights_path)
        model.trainable = False
        return model

    def data_labels_split(self):
        x_data, y_data = [], []
        for i in range(len(self.val_gen)):
            x_batch, y_batch = self.val_gen[i]
            x_data.append(x_batch)
            y_data.append(y_batch)

        x_all = np.vstack(x_data)
        y_all = np.concatenate(y_data)

        class_0_idx = np.where(y_all == 0)[0]
        class_1_idx = np.where(y_all == 1)[0]

        return x_all, y_all, class_0_idx, class_1_idx

    def get_all_features_labels(self, encoder_model):
        features, labels = [], []
        for i in range(len(self.val_gen)):
            x_batch, y_batch = self.val_gen[i]
            feats = encoder_model.predict(x_batch, verbose=0)
            features.append(feats)
            labels.append(y_batch)

        return np.vstack(features), np.concatenate(labels)

    def cross_validate_on_features(self, features, labels, x_raw, y_raw, class_indices,
                                   num_per_class=10, save_dir='avg_subset_output'):
        save_dir += '_' + str(num_per_class)
        n_runs = int(16000 / (2 * num_per_class))
        os.makedirs(save_dir, exist_ok=True)
        acc_list = []
        subset_indices = []

        class_0_idx, class_1_idx = class_indices

        for run in range(n_runs):
            idx_0 = np.random.choice(class_0_idx, num_per_class, replace=False)
            idx_1 = np.random.choice(class_1_idx, num_per_class, replace=False)
            train_idx = np.concatenate([idx_0, idx_1])

            x_train = features[train_idx]
            y_train = labels[train_idx]

            clf = LogisticRegression(max_iter=2000)
            clf.fit(x_train, y_train)

            val_idx = list(set(range(len(features))) - set(train_idx))
            x_val = features[val_idx]
            y_val = labels[val_idx]

            preds = clf.predict(x_val)
            acc = accuracy_score(y_val, preds)
            acc_list.append(acc)
            subset_indices.append(train_idx)

        avg_acc = np.mean(acc_list)
        best_diff = float('inf')
        best_idx = -1
        for i, acc in enumerate(acc_list):
            if abs(acc - avg_acc) < best_diff:
                best_diff = abs(acc - avg_acc)
                best_idx = i

        idx_to_save = subset_indices[best_idx]
        for i, idx in enumerate(idx_to_save):
            label = int(labels[idx])
            out_dir = os.path.join(save_dir, str(label))
            os.makedirs(out_dir, exist_ok=True)
            save_img(os.path.join(out_dir, f"img_{i}.png"), x_raw[idx])

        print(f"Average accuracy over {n_runs} runs: {avg_acc:.4f}")
        print(f"Saved representative subset (run {best_idx}) to: {save_dir}")


parser = argparse.ArgumentParser(description="Cross validate few-shot classifier performance and pick average input subset")
parser.add_argument('--encoder', type=str, default='../models/encoder_classifier_weights.h5', help='Path to the pretrained encoder weights in .h5')
parser.add_argument('--dataset', type=str, default=GLOBAL_DATASET_PATH, help='Path to the PCAM dataset')
parser.add_argument('--num_per_class', type=int, default=32, help='Number instances per class')
parser.add_argument('--output', type=str, default='avg_subset_output', help='Output directory')
args = parser.parse_args()

selector = FewShotSubsetSelector(base_dir=args.dataset)
encoder = selector.get_frozen_encoder(weights_path=args.encoder)
x_raw, y_raw, class_0_idx, class_1_idx = selector.data_labels_split()
features, labels = selector.get_all_features_labels(encoder)
selector.cross_validate_on_features(features, labels, x_raw, y_raw, [class_0_idx, class_1_idx], num_per_class=args.num_per_class, save_dir=args.output)