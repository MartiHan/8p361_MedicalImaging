from tensorflow.keras import layers
import tensorflow as tf

class RandomMasking(tf.keras.layers.Layer):
    def __init__(self, patch_size=8, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def call(self, images):
        B, H, W, C = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[3]
        ph, pw = self.patch_size, self.patch_size

        num_patches_h = H // ph
        num_patches_w = W // pw
        num_patches = num_patches_h * num_patches_w
        num_mask = tf.cast(tf.cast(num_patches, tf.float32) * self.mask_ratio, tf.int32)

        def mask_single(img):
            patches = tf.image.extract_patches(
                images=tf.expand_dims(img, 0),
                sizes=[1, ph, pw, 1],
                strides=[1, ph, pw, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )  # shape: (1, H//ph, W//pw, ph*pw*C)

            N = tf.shape(patches)[1] * tf.shape(patches)[2]
            idx = tf.range(N)
            shuffled = tf.random.shuffle(idx)[:num_mask]

            # Create binary mask for patches
            mask_flat = tf.ones(N, dtype=tf.float32)
            mask_flat = tf.tensor_scatter_nd_update(mask_flat, tf.expand_dims(shuffled, 1), tf.zeros_like(shuffled, dtype=tf.float32))
            mask = tf.reshape(mask_flat, (num_patches_h, num_patches_w, 1))

            # Upsample mask to original image size
            mask = tf.image.resize(mask, [H, W], method='nearest')  # shape: (H, W, 1)
            return img * mask  # mask is 0 where we hide

        return tf.map_fn(mask_single, images)
