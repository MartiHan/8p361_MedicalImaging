import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import model_from_json, Model
from pcam_loader.pcam_loader import PCAMDataLoader


class LatentSpaceVisualizer:
    def __init__(self, base_dir, json_path, weights_path, input_shape=(96, 96, 3)):
        self.base_dir = base_dir
        self.json_path = json_path
        self.weights_path = weights_path
        self.input_shape = input_shape
        self.model = None
        self.encoder_model = None

    def load_model(self):
        with open(self.json_path, 'r') as f:
            model_json = f.read()
        self.model = model_from_json(model_json)
        self.model.load_weights(self.weights_path)
        #self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Extract latent representation from the 'gap' layer
        self.encoder_model = Model(inputs=self.model.input,
                                   outputs=self.model.get_layer('gap').output)


    def prepare_data(self):
        loader = PCAMDataLoader(base_dir=self.base_dir, image_size=self.input_shape[0])
        _, val_gen = loader.get_generators()
        return val_gen

    def compute_tsne(self):
        val_gen = self.prepare_data()
        features = self.encoder_model.predict(val_gen, verbose=1)
        labels = val_gen.classes

        tsne = TSNE(n_components=2, random_state=42, perplexity=50)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(8, 6))

        metastases_idx = labels == 1
        no_metastases_idx = labels == 0

        plt.scatter(tsne_results[metastases_idx, 0], tsne_results[metastases_idx, 1],
                    c='red', label='Metastases', alpha=0.5)

        plt.scatter(tsne_results[no_metastases_idx, 0], tsne_results[no_metastases_idx, 1],
                    c='blue', label='No metastases', alpha=0.5)

        plt.legend()
        plt.title("t-SNE of Feature Extractor Latent Space")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.grid(True)
        plt.show()


visualizer = LatentSpaceVisualizer(
    base_dir='../../../../Datasets',
    json_path='../models/gradcam_stain_standardization.json',
    weights_path='../models/gradcam_stain_standardization.hdf5'
)
visualizer.load_model()
visualizer.compute_tsne()
