import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from pcam_loader.pcam_loader import PCAMDataLoader


class FewShotTrainer:
    def __init__(self, base_dir, model_name='few_shot_classifier', image_size=96, batch_size=32):
        self.base_dir = base_dir
        self.model_name = model_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        self.train_gen, self.val_gen = self._load_data()

    def _load_data(self):
        loader = PCAMDataLoader(base_dir=self.base_dir, image_size=self.image_size)
        return loader.get_generators(train_batch_size=self.batch_size, val_batch_size=self.batch_size, train_val=False)

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

    def get_frozen_encoder(self, input_shape=(96, 96, 3), weights_path=None):
        inputs = Input(shape=input_shape)
        x = self.encoder_blocks(inputs)
        model = models.Model(inputs, x, name="frozen_encoder")

        if weights_path:
            model.load_weights(weights_path)
        model.trainable = False
        return model

    def get_few_shot_classifier(self, input_shape=(96, 96, 3), weights_path=None):
        encoder = self.get_frozen_encoder(input_shape, weights_path)
        inputs = Input(shape=input_shape)
        x = encoder(inputs)
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        x = layers.Dense(128, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.5, name='dropout')(x)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        model = models.Model(inputs, outputs, name="few_shot_classifier")
        return model

    def get_few_shot_subset(self, generator, num_per_class=10):
        x_few_shot, y_few_shot = [], []
        class_counts = {0: 0, 1: 0}
        total_needed = num_per_class * 2

        for i in range(len(generator)):
            x_batch, y_batch = generator[i]
            for x, y in zip(x_batch, y_batch):
                y = int(y)
                if class_counts[y] < num_per_class:
                    x_few_shot.append(x)
                    y_few_shot.append(y)
                    class_counts[y] += 1
                if sum(class_counts.values()) == total_needed:
                    return np.array(x_few_shot), np.array(y_few_shot)

        raise ValueError("Not enough samples to build the few-shot subset.")

    def train(self, num_per_class=8000, epochs=3, batch_size=32, weights_path=None):
        self.model = self.get_few_shot_classifier(weights_path=weights_path)
        x_few, y_few = self.get_few_shot_subset(self.val_gen, num_per_class=num_per_class)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        model_filepath = self.model_name + '.json'
        weights_filepath = self.model_name + '_weights.hdf5'

        model_json = self.model.to_json()
        with open(model_filepath, 'w') as json_file:
            json_file.write(model_json)

        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', save_best_only=False, mode='min')
        tensorboard = TensorBoard(log_dir=os.path.join('logs', self.model_name))
        callbacks_list = [checkpoint, tensorboard]

        history = self.model.fit(x_few, y_few, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
        return history


trainer = FewShotTrainer(base_dir='avg_subset_output_10')
trainer.train(num_per_class=10, epochs=50, batch_size=32, weights_path='../models/ranking_encoder_weights.h5')
