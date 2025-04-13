from tensorflow.keras import layers
import tensorflow as tf

class PatchEmbed(layers.Layer):
    def __init__(self, patch_size=8, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)

    def call(self, x):
        x = self.proj(x)  # (B, H/P, W/P, embed_dim)
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.embed_dim))  # (B, num_patches, embed_dim)
        return x
