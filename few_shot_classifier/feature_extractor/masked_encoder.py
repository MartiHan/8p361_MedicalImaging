from tensorflow.keras import layers, Model, Input
from pcam_loader.pcam_loader import PCAMDataLoader
import tensorflow as tf
from feature_extractor.patch_embedding import PatchEmbed
from feature_extractor.random_masking import RandomMasking
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os

class MaskedAutoEncoder:
    def __init__(self, input_shape=(96, 96, 3), encoder_freeze=False, weights=None):
        self.input_shape = input_shape
        self.encoder_freeze = encoder_freeze
        self.weights = weights

    def build_encoder(self, num_patches=144, embed_dim=128, depth=4):
        inputs = Input(shape=self.input_shape, name='encoder_input')
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

        model = models.Model(inputs=inputs, outputs=x, name='encoder')
        return model

    def build_decoder(self, output_shape=(96, 96, 3)):
        inputs = Input(shape=(12, 12, 64))  # Output of encoder (pooled 3 times from 96 → 12)

        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(inputs)  # 12→24
        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)  # 24→48
        x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)  # 48→96
        x = layers.Conv2D(3, 1, activation='sigmoid')(x)

        return models.Model(inputs, x, name='decoder')

    def build_mae(self, input_shape=(96, 96, 3), patch_size=8, embed_dim=128, mask_ratio=0.75):
        inputs = Input(shape=input_shape)

        # Apply masking
        masked = RandomMasking(mask_ratio=mask_ratio)(inputs)

        # Encoder (your model)
        encoder = self.build_encoder(input_shape)
        encoded = encoder(masked)  # output shape (B, 12, 12, 64)

        # Decoder
        decoder = self.build_decoder(output_shape=input_shape)
        reconstructed = decoder(encoded)

        return models.Model(inputs, reconstructed, name='mae_with_custom_encoder')


class ModelTrainer:
    def __init__(self, base_dir, input_shape=(96, 96, 3)):
        self.base_dir = base_dir
        self.input_shape = input_shape
        self.model = None

    def prepare_data(self):
        loader = PCAMDataLoader(base_dir=self.base_dir, image_size=self.input_shape[0], standardize=False)
        return loader.get_generators(class_mode=None)

    def build_model(self):
        encoder = MaskedAutoEncoder(input_shape=self.input_shape)
        self.model = encoder.build_mae()

        self.model.compile(optimizer='adam', loss='mse')

    def self_supervised_wrapper(self, generator):
        for batch in generator:
            yield batch, batch

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
            self.self_supervised_wrapper(train_gen), epochs=epochs, steps_per_epoch=train_steps,
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
trainer.train(val_gen, epochs=50)

