import os
import base64
import io
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set dataset directory
FALSE_POSITIVES_PATH = "false_positives/"

# Load Model Architecture and Weights
with open("my_first_cnn_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("my_first_cnn_model_weights.hdf5")

# Function to Get the Last Conv Layer
def get_last_conv_layer(model):
    """Finds the last convolutional layer in the CNN."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model.")

def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlays the Grad-CAM heatmap on the original image with adjustable opacity (alpha)."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.uint8(255 * image)  # Convert original image to uint8
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)  # ✅ Dynamic Alpha
    return superimposed_img

# Function to preprocess images
def preprocess_image(image_path, target_size=(96, 96)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    return img_array

# Load all images from the false positives folder
def load_images():
    if not os.path.exists(FALSE_POSITIVES_PATH):
        raise FileNotFoundError(f"⚠ Folder '{FALSE_POSITIVES_PATH}' does not exist! Run the filtering script first.")

    image_list = []
    for filename in sorted(os.listdir(FALSE_POSITIVES_PATH)):  # Sorted by confidence
        img_path = os.path.join(FALSE_POSITIVES_PATH, filename)
        img_array = preprocess_image(img_path)
        image_list.append((img_array, filename))

    if not image_list:
        raise ValueError("⚠ No images found in 'false_positives/'. Run the filtering script first!")

    return image_list

# Load images
image_list = load_images()
num_images = len(image_list)

# Function to Compute Grad-CAM Heatmap (Fixed)
def compute_grad_cam(model, image, layer_name=None):
    """Generates a Grad-CAM heatmap for the given image."""
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

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = np.mean(conv_output * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap

# Function to Convert Image to Base64
def img_to_base64(img_array, size=(96, 96), title=""):
    fig, ax = plt.subplots(figsize=(size[0] / 32, size[1] / 32))
    ax.imshow(img_array)
    ax.axis("off")
    buf = io.BytesIO()
    plt.title(title, color="white")
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#343a40")
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
app.config.suppress_callback_exceptions = True  # Allow dynamic callback components

# Default Dark Mode Styling
dark_theme = {
    "backgroundColor": "#343a40",
    "color": "#ffffff",
    "minHeight": "100vh",
    "width": "100%",
    "margin": "0",
    "padding": "0",
}

# App Layout
app.layout = html.Div(
    [
        # Full-width Navbar
        html.H1(
            "PCam Analyzer", className="bg-primary text-white p-4 mb-2 text-center"
        ),

        # Store the current index of the displayed image
        dcc.Store(id="current-index", data=0),
        dcc.Interval(id="startup-trigger", interval=1, max_intervals=1),

        # Image Display Area with Label
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            [
                                html.H5(id="image-label", className="text-center mt-2"),
                                html.H6(id="confidence-score", className="text-center mt-2"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Img(id="output-image",
                                                     style={"maxWidth": "300px", "borderRadius": "0px", "paddingTop": "0px"}),
                                            width="auto", className="text-center"
                                        ),
                                        dbc.Col(
                                            children=[
                                                html.Img(id="gradcam-image",
                                                         style={"maxWidth": "300px", "borderRadius": "0px", "paddingTop": "35px"}),

                                                dcc.Slider(
                                                    id="opacity-slider",
                                                    min=0, max=1, step=0.05,
                                                    value=0.5,  # Default opacity
                                                    marks={0: "0", 0.5: "0.5", 1: "1"},
                                                    tooltip={"placement": "bottom"},
                                                ),
                                            ],
                                        width="auto",
                                        className="text-center"),
                                    ],
                                    className="d-flex justify-content-center align-items-center"
                                ),

                            ],
                            className="text-center",
                        ),
                        width=12,
                    )
                ),

                dbc.Row(
                    style={"padding": "10px"},
                    children=[
                        dbc.Col(
                            dbc.Button("Flag the image", color="danger", style={"width": "300px"}),
                            className="mx-auto text-center",
                        )
                    ]
                ),

                # Thumbnail Preview Row
                dbc.Row(
                    dbc.Col(
                        html.Div(id="thumbnail-gallery", className="d-flex justify-content-center mt-3"),
                        width=12,
                    )
                ),

                # Navigation Buttons (Below the Image)
                dbc.Row(
                    dbc.Col(
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Previous", id="prev-btn", n_clicks=0, type="button", color="primary", className="me-1",
                                           style={"width": "100px"}),
                                dbc.Button("Next", id="next-btn", n_clicks=0, color="primary", className="me-1", style={"width": "100px"}),
                            ],
                            className="mt-3 d-flex justify-content-center",
                        ),
                        width=4,  # Reduce width to make buttons smaller and centered
                        className="mx-auto text-center",
                    )
                ),
            ],
            fluid=True,
        ),
    ],
    style=dark_theme,
)

# Callback to Handle Navigation & Thumbnail Updates
@app.callback(
    [Output("output-image", "src"),
     Output("gradcam-image", "src"),
     Output("image-label", "children"),
     Output("confidence-score", "children"),
     Output("thumbnail-gallery", "children"),
     Output("current-index", "data")],
    [Input("startup-trigger", "n_intervals"),
     Input("prev-btn", "n_clicks"),
     Input("next-btn", "n_clicks"),
     Input({"type": "thumb", "index": ALL}, "n_clicks"),
     Input("opacity-slider", "value")],
    [State("current-index", "data")]
)
def update_image(startup, prev_clicks, next_clicks, thumb_clicks, opacity, current_index):
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "prev-btn":
        current_index = max(0, current_index - 1)
    elif triggered_id == "next-btn":
        current_index = min(num_images - 1, current_index + 1)
    elif "thumb" in triggered_id:
        current_index = int(eval(triggered_id)["index"])

    img_array, filename = image_list[current_index]
    main_img_src = img_to_base64(img_array, size=(96, 96), title="Original")

    # Compute Grad-CAM Heatmap
    heatmap = compute_grad_cam(loaded_model, img_array)
    gradcam_overlay = overlay_heatmap(img_array, heatmap, opacity)

    # Extract Confidence Score from Filename
    confidence = 100 - int(filename.split("_")[0])  # Extract confidence
    confidence_text = f"Metastases presence confidence: {confidence} %"

    # **Sliding Window for Thumbnails**
    half_window = 5
    start_idx = max(0, current_index - half_window)
    end_idx = min(num_images, current_index + half_window + 1)

    if end_idx - start_idx < 11:
        if start_idx == 0:
            end_idx = min(num_images, 11)
        elif end_idx == num_images:
            start_idx = max(0, num_images - 11)

    thumb_indexes = list(range(start_idx, end_idx))

    # Generate Thumbnail Gallery
    thumbnails = [
        html.Img(
            src=img_to_base64(image_list[i][0], size=(64, 64)),
            style={
                "width": "64px",
                "height": "64px",
                "border": f"3px solid {'yellow' if i == current_index else 'green'}",
                "cursor": "pointer",
                "margin": "5px"
            },
            id={"type": "thumb", "index": i}
        )
        for i in thumb_indexes
    ]

    label = "Given label: No metastases"
    return main_img_src, img_to_base64(gradcam_overlay, size=(96,96), title="GradCAM"), label, confidence_text, thumbnails, current_index

if __name__ == "__main__":
    app.run_server(debug=True)
