import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from sklearn.metrics import roc_curve, auc
from pcam_loader.pcam_loader import PCAMDataLoader


class ModelEvaluator:
    def __init__(self, base_dir, json_path, weights_path, input_shape=(96, 96, 3)):
        self.base_dir = base_dir
        self.json_path = json_path
        self.weights_path = weights_path
        self.input_shape = input_shape
        self.model = None

    def load_model(self):
        with open(self.json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.weights_path)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def prepare_data(self):
        loader = PCAMDataLoader(base_dir=self.base_dir, image_size=self.input_shape[0])
        _, val_gen = loader.get_generators()
        return val_gen

    def evaluate(self):
        val_gen = self.prepare_data()
        predictions = self.model.predict(val_gen, verbose=1)
        y_true = val_gen.classes
        fpr, tpr, _ = roc_curve(y_true, predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

        return roc_auc

evaluator = ModelEvaluator(
    base_dir='/home/martina/Documents/Projects/8P361 AI Project for Medical Imaging/Datasets/',
    json_path='custom_test.json',
    weights_path='custom_test_weights.hdf5'
)
evaluator.load_model()
auc_score = evaluator.evaluate()
print(f"AUC Score: {auc_score:.4f}")
