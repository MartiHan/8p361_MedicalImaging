from tensorflow.keras import layers, Model, Input
from pcam_loader.pcam_loader import PCAMDataLoader
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os

class MaskedAutoEncoder:
    def __init__(self, input_shape=(96, 96, 3), encoder_freeze=False, weights=None):
        self.input_shape = input_shape
        self.encoder_freeze = encoder_freeze
        self.weights = weights

    def build(self):
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 4 (Final Convolutional Block for Grad-CAM)
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='gradcam_layer'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            # Fully Connected Layer
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])

        # compile the model
        #model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        return model

class ModelTrainer:
    def __init__(self, base_dir, input_shape=(96, 96, 3)):
        self.base_dir = base_dir
        self.input_shape = input_shape
        self.model = None

    def prepare_data(self):
        loader = PCAMDataLoader(base_dir=self.base_dir, image_size=self.input_shape[0])
        return loader.get_generators()

    def build_model(self):
        encoder = MaskedAutoEncoder(input_shape=self.input_shape)
        self.model = encoder.build()

        # Add a simple classifier head for binary classification
        x = layers.GlobalAveragePooling2D()(self.model.output[0])
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        self.model = Model(self.model.input, output)
        self.model.compile(Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_gen, epochs=10, batch_size=32):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model_name = 'custom_test'
        model_filepath = model_name + '.json'
        weights_filepath = model_name + '_weights.hdf5'

        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, mode='min')
        tensorboard = TensorBoard(os.path.join('logs', model_name))
        callbacks_list = [checkpoint, tensorboard]

        train_steps = train_gen.n // train_gen.batch_size
        val_steps = val_gen.n // val_gen.batch_size

        history = self.model.fit(
            train_gen, epochs=epochs, steps_per_epoch=train_steps,
            callbacks=callbacks_list
        )

        # Save model architecture to JSON
        model_json = self.model.to_json()
        with open(model_filepath, "w") as json_file:
            json_file.write(model_json)

        # Save model weights to HDF5
        #self.model.save_weights("u_net_Test_weights.hdf5")

        return history


trainer = ModelTrainer(base_dir='/home/martina/Documents/Projects/8P361 AI Project for Medical Imaging/Datasets/')
train_gen, val_gen = trainer.prepare_data()
trainer.build_model()


trainer.train(val_gen, epochs=1)

