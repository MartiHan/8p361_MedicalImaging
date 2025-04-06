import numpy as np
import tensorflow as tf
import cv2

def get_last_conv_layer(model):
    """Finds the last convolutional layer in the CNN."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model.")

def compute_grad_cam(model, image, layer_name=None):
    """Generates a Grad-CAM heatmap for the given image."""
    print("Selected layer: ", layer_name)
    if layer_name is None:
        layer_name = get_last_conv_layer(model)

    img_tensor = np.expand_dims(image, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs[0],
        outputs=[model.get_layer(layer_name).output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor)
        class_index = int(np.argmax(predictions))  # ✅ FIXED Invalid Index Error
        class_output = predictions[:, class_index]

    predicted_prob = model.predict(img_tensor)[0][0]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = np.mean(conv_output * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    print("Predicted Probability: ", predicted_prob)
    return heatmap


def compute_hires_cam(model, image, class_index=0, layer_name=None):
    """
    Generates a HiResCAM heatmap for a given image.
    - model: Trained CNN model.
    - image: Input image (shape: (96, 96, 3)).
    - class_index: Index of the predicted class (default: 0 for binary classification).
    - layer_name: Name of the convolutional layer to visualize.

    Returns:
    - High-resolution CAM heatmap.
    """
    if layer_name is None:
        # Automatically find the last Conv2D layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    # Expand dimensions for model input
    img_tensor = np.expand_dims(image, axis=0)

    # Create model that outputs feature maps + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor)
        class_output = predictions[:, class_index]

    # Compute gradients of the class output w.r.t. conv layer
    grads = tape.gradient(class_output, conv_output)

    # Only keep **positive** gradients (HiResCAM technique)
    positive_grads = tf.maximum(grads, 0)  # Removes negative contributions

    # Compute importance weights
    pooled_grads = tf.reduce_mean(positive_grads, axis=(0, 1, 2))

    # Apply weights to feature maps
    conv_output = conv_output[0]
    hires_cam = np.mean(conv_output * pooled_grads, axis=-1)

    # Normalize heatmap
    hires_cam = np.maximum(hires_cam, 0)
    hires_cam /= np.max(hires_cam)

    return hires_cam

def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlays the Grad-CAM heatmap on the original image with adjustable opacity (alpha)."""
    #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

        # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize heatmap (0 to 1)
    heatmap = np.clip(heatmap, 0, 1)

    # Convert heatmap to uint8 (0-255) for colormap application
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply colormap (JET for better visibility)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Dynamically scale alpha (transparency) based on activation strength
    alpha_map = 0 + alpha * heatmap  # Scale between alpha_min and alpha_max
    alpha_map = np.expand_dims(alpha_map, axis=-1)  # Make it (H, W, 1)
    alpha_map = np.repeat(alpha_map, 3, axis=-1)  # Match RGB channels

    # Blend dynamically based on the alpha map
    blended = (image * (1 - alpha_map) + heatmap_colored * alpha_map).astype(np.uint8)

    return blended

def compute_xgrad_cam(model, image, class_idx=0, last_conv_layer_name="conv2d"):
    """
    Computes xGrad-CAM heatmap (an improved Grad-CAM with second-order gradients).

    - `model`: Trained CNN model.
    - `image`: Input image in shape (96, 96, 3).
    - `class_idx`: Target class index for visualization.
    - `last_conv_layer_name`: Name of the last convolutional layer.

    Returns:
    - xGrad-CAM heatmap (96, 96).
    """

    #  Ensure correct input shape
    if image.shape != (96,96,3):
        raise ValueError(f"Input image must be (96,96,3), got {image.shape}")

    #  Expand dimensions for batch processing
    image_batch = np.expand_dims(image, axis=0).astype(np.float32)

    #  Extract last convolutional layer and model output
    last_conv_layer = model.get_layer(last_conv_layer_name).output
    output_layer = model.output

    #  Create new model mapping input → feature maps + prediction
    xgrad_model = tf.keras.Model(inputs=model.input, outputs=[last_conv_layer, output_layer])

    #  Compute gradients using GradientTape
    with tf.GradientTape() as tape:
        conv_output, predictions = xgrad_model(image_batch)
        class_output = predictions[:, class_idx]  # Extract class prediction

    #  Compute **first-order gradients** (Regular Grad-CAM)
    grads = tape.gradient(class_output, conv_output)

    #  Compute **second-order gradients** (New in xGrad-CAM)
    with tf.GradientTape() as tape2:
        tape2.watch(conv_output)
        second_order_output = tf.reduce_sum(conv_output * grads, axis=(1, 2))  # Aggregate gradients
    second_grads = tape2.gradient(second_order_output, conv_output)

    #  Compute xGrad-CAM activation map
    weighted_features = np.sum(second_grads[0] * conv_output[0], axis=-1)

    #  Normalize the heatmap
    weighted_features = np.maximum(weighted_features, 0)  # ReLU to remove negatives
    weighted_features /= np.max(weighted_features) if np.max(weighted_features) != 0 else 1  # Normalize

    #  Resize heatmap to match original image
    heatmap = cv2.resize(weighted_features, (96, 96))

    return heatmap
