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
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import visualkeras
from PIL import ImageFont
from PIL import Image
from keras.utils import plot_model


# Set dataset directory
FALSE_POSITIVES_PATH = "marek_gpu/true_positives"

DEFAULT_DATASET="basic_cnn_model.json"
DEFAULT_WEIGHT="basic_model_weights.hdf5"

# Load Model Architecture and Weights
with open(DEFAULT_DATASET, "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(DEFAULT_WEIGHT)
print(loaded_model.summary())



def generate_model_visualization(model):
    """Generates a visual representation of the CNN model and saves it."""
    # font = ImageFont.truetype("arial.ttf", 12)  # Set font
    visualkeras.layered_view(model, legend=True, to_file="assets/model_architecture.png")  # Save image

    plot_model(model, to_file='assets/model_plot.png', show_shapes=True, show_layer_names=True)

    return "model_architecture.png"

generate_model_visualization(loaded_model)

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
app.title = "PCam Analyzer"
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
        dcc.Store(id="visited-images", data=[]),
        dcc.Store(id="flagged-samples", data=[]),
        dcc.Interval(id="startup-trigger", interval=1, max_intervals=1),

        # Image Display Area with Label
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                        html.H2("Model Architecture", className="text-center mt-2"),
                                        dbc.Row(
                                            [
                                                dbc.Col([
                                                    html.Img(id="model-visual", src="assets/model_architecture.png",
                                                             style={"maxWidth": "370px"}),
                                                    html.Img(id="model-visual3", src="assets/confusion_matrix.png",
                                                             style={"maxHeight": "322px", "paddingTop": "15px"}),
                                                ],
                                                    style={"padding:": "15px"},
                                                    className="p-0 m-0",
                                                    width=5,
                                                ),
                                                html.Div(style={"width": "10px"}),
                                                dbc.Col([
                                                    dbc.Row([
                                                        dbc.Col(
                                                            html.Img(id="model-visual2", src="assets/model_plot.png", style={"maxHeight": "410px"}),
                                                            className="p-0 m-0",
                                                        ),
                                                        html.Div(style={"width": "10px"}),
                                                        dbc.Col(
                                                            [
                                                            html.H6("Loaded model: "),
                                                            html.Label(f"{DEFAULT_DATASET}"),
                                                            dcc.Upload(
                                                                id="upload-image",
                                                                children=html.Div(
                                                                    ["Replace the Model"]
                                                                ),
                                                                style={
                                                                    "width": "100%",
                                                                    "height": "100px",
                                                                    "lineHeight": "100px",
                                                                    "borderWidth": "2px",
                                                                    "borderStyle": "dashed",
                                                                    "borderRadius": "10px",
                                                                    "textAlign": "center",
                                                                    "margin": "10px",
                                                                },
                                                                multiple=False,
                                                            ),
                                                            html.Div(style={"paddingTop": "10px"}),
                                                            html.H6("Loaded weights: "),
                                                            html.Label(f"{DEFAULT_WEIGHT}"),

                                                            dcc.Upload(
                                                                id="upload-weight",
                                                                children=html.Div(
                                                                    ["Replace the Weights"]
                                                                ),
                                                                style={
                                                                    "width": "100%",
                                                                    "height": "100px",
                                                                    "lineHeight": "100px",
                                                                    "borderWidth": "2px",
                                                                    "borderStyle": "dashed",
                                                                    "borderRadius": "10px",
                                                                    "textAlign": "center",
                                                                    "margin": "10px",
                                                                },
                                                                multiple=False,
                                                            ),
                                                            dbc.Row(
                                                                style={"paddingLeft": "7px"},
                                                                children=[
                                                                    dbc.Col(
                                                                        dbc.Button("Regenerate the Classifier",
                                                                                   color="primary",
                                                                                   style={"width": "200px"}),
                                                                        #className="mx-auto text-center",
                                                                        width=2,
                                                                    )
                                                                ]
                                                            ),

                                                            ],
                                                            className="p-0 m-0",
                                                        ),
                                                        ],
                                                            className="g-0",
                                                    ),
                                                    html.Img(id="model-visual4", src="assets/histogram.png",
                                                                 style={"maxWidth": "490px", "paddingTop": "15px"}),
                                                ],
                                                    className="p-0 m-0",
                                                    width=6,
                                                )
                                                #html.Div(style={"width": "10px"}),

                                            ],
                                                className="g-0"
                                        ),
                                        #html.Div(style={"width": "10px"}),

                            ],
                            #width=4,  # Set column width
                        ),
                        dbc.Col(
                            [
                            html.H2("False Positive Samples Preview", className="text-center mt-2"),
                            dbc.Row(
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H5(id="image-label", className="text-center mt-2"),
                                            html.H6(id="confidence-score", className="text-center mt-2"),
                                            dbc.Select(
                                                id="gradcam-layer-dropdown",
                                                options=[
                                                    {"label": layer.name, "value": layer.name} for layer in loaded_model.layers if "conv" in layer.name
                                                ],
                                                value=get_last_conv_layer(loaded_model),  # Default: Last Conv Layer
                                                #clearable=False,
                                                style={"width": "20%", "margin-bottom": "10px"},
                                                className="mx-auto text-center",
                                            ),
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
                                                    dbc.Col(
                                                        html.Img(id="hircam-image",
                                                                 style={"maxWidth": "300px", "borderRadius": "0px", "paddingTop": "0px"}),
                                                        width="auto", className="text-center"
                                                    ),

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
                                        dbc.Button("Flag the image", id="flag-btn", color="danger", style={"width": "300px"}),
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

                                dbc.Row(
                                    style={"paddingTop": "30px"},
                                    children=[
                                        dbc.Col(
                                            dbc.Button("Export 0 Flagged Images", id="export-btn", color="warning", style={"width": "500px"}),
                                            className="mx-auto text-center",
                                        )
                                    ]
                                ),
                            ]
                        )
                    ]
                )
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
     Output("hircam-image", "src"),
     Output("image-label", "children"),
     Output("confidence-score", "children"),
     Output("thumbnail-gallery", "children"),
     Output("current-index", "data"),
     Output("flagged-samples", "data"),
     Output("visited-images", "data"),
     Output("export-btn", "children"),
     ],
    [Input("startup-trigger", "n_intervals"),
     Input("prev-btn", "n_clicks"),
     Input("next-btn", "n_clicks"),
     Input("flag-btn", "n_clicks"),
     Input({"type": "thumb", "index": ALL}, "n_clicks"),
     Input("opacity-slider", "value"),
     Input("gradcam-layer-dropdown", "value")],
    [State("current-index", "data"),
     State("flagged-samples", "data"),
     State("visited-images", "data")]
)
def update_image(startup, prev_clicks, next_clicks, flag_btn, thumb_clicks, opacity, selected_layer, current_index, flagged_samples, visited_images):
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
    elif triggered_id == "flag-btn":
        filename = image_list[current_index][1]  # Get current filename
        if filename not in flagged_samples:
            flagged_samples.append(filename)  # ✅ Save filename of flagged image

    filename = image_list[current_index][1]
    if filename not in visited_images:
        visited_images.append(filename)  # ✅ Mark as visited

    img_array, filename = image_list[current_index]
    main_img_src = img_to_base64(img_array, size=(96, 96), title="Original")

    # Compute Grad-CAM Heatmap
    heatmap = compute_grad_cam(loaded_model, img_array, layer_name=selected_layer)
    gradcam_overlay = overlay_heatmap(img_array, heatmap, opacity)

    heatmap = compute_hires_cam(loaded_model, img_array, layer_name=selected_layer)
    hircam_overlay = overlay_heatmap(img_array, heatmap, opacity)

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
                "border": f"3px solid {'red' if image_list[i][1] in flagged_samples else ('yellow' if i == current_index else ('green' if image_list[i][1] in visited_images else 'white'))}",
                "cursor": "pointer",
                "margin": "5px"
            },
            id={"type": "thumb", "index": i}
        )
        for i in thumb_indexes
    ]

    flagged_count = len(flagged_samples)
    label = "Given label: No metastases"
    return (main_img_src, img_to_base64(gradcam_overlay, size=(96,96), title="GradCAM"),
            img_to_base64(hircam_overlay, size=(96,96), title="HiResCAM"), label, confidence_text, thumbnails,
            current_index, flagged_samples, visited_images, f"Export {flagged_count} Flagged Images")

if __name__ == "__main__":
    app.run_server(debug=True)
