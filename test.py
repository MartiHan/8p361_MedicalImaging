import os
import base64
import io
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Image Dataset Parameters
IMAGE_SIZE = 96
CATEGORY_SIZE = 10  # Number of images to load per class
DATASET_PATH = '/home/martina/Documents/University/Year 2/Q3/8P361 AI Project for Medical Imaging/Datasets/'  # Update this path

# Function to Load Images
def get_pcam_generators(base_dir, batch_size=32):
    train_path = os.path.join(base_dir, 'train+val', 'train')

    RESCALING_FACTOR = 1.0 / 255
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

    train_gen = datagen.flow_from_directory(
        train_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='binary'
    )
    return train_gen

# Get Data Generator
train_gen = get_pcam_generators(DATASET_PATH)

# Load and Preprocess Images
class_names = ['Benign', 'Malignant']
selected_images = {0: [], 1: []}

while len(selected_images[0]) < CATEGORY_SIZE or len(selected_images[1]) < CATEGORY_SIZE:
    images, labels = next(train_gen)
    for img, label in zip(images, labels):
        label_int = int(label)
        if len(selected_images[label_int]) < CATEGORY_SIZE:
            selected_images[label_int].append((img, label_int))  # Store image with label
    if len(selected_images[0]) >= CATEGORY_SIZE and len(selected_images[1]) >= CATEGORY_SIZE:
        break

# Combine images into one list
image_list = selected_images[0] + selected_images[1]

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

        # Image Display Area with Label
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            [
                                html.H5(id="image-label", className="text-center mt-2"),
                                html.Img(
                                    id="output-image",
                                    style={"maxWidth": "300px", "borderRadius": "10px", "display": "block", "margin": "auto"},
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

# Function to Convert Image to Base64
def img_to_base64(img_array, size=(96, 96)):
    fig, ax = plt.subplots(figsize=(size[0] / 32, size[1] / 32))
    ax.imshow(img_array)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#343a40")
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

# Callback to Handle Navigation & Thumbnail Clicks
@app.callback(
    [Output("output-image", "src"),
     Output("image-label", "children"),
     Output("thumbnail-gallery", "children"),
     Output("current-index", "data")],
    [Input("prev-btn", "n_clicks"),
     Input("next-btn", "n_clicks"),
     Input({"type": "thumb", "index": ALL}, "n_clicks")],
    [State("current-index", "data")]
)
def update_image(prev_clicks, next_clicks, thumb_clicks, current_index):
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Determine new index
    if triggered_id == "prev-btn":
        current_index = (current_index - 1) % len(image_list)
    elif triggered_id == "next-btn":
        current_index = (current_index + 1) % len(image_list)
    elif "thumb" in triggered_id:
        current_index = int(eval(triggered_id)["index"])

    # Get the current main image and label
    img_array, img_label = image_list[current_index]
    main_img_src = img_to_base64(img_array, size=(96, 96))
    label_text = "Benign" if img_label == 0 else "Malignant"

    # Get 5 previous and 5 next thumbnails
    prev_thumbs = [(current_index - i - 1) % len(image_list) for i in range(5)]
    next_thumbs = [(current_index + i + 1) % len(image_list) for i in range(5)]
    thumb_indexes = prev_thumbs[::-1] + [current_index] + next_thumbs  # Ordered list of thumbnails

    # Generate Thumbnail Gallery
    thumbnails = [
        html.Img(
            src=img_to_base64(image_list[i][0], size=(64, 64)),
            style={
                "width": "64px",
                "height": "64px",
                "border": f"3px solid {'yellow' if i == current_index else ('green' if image_list[i][1] == 0 else 'red')}",
                "cursor": "pointer",
                "margin": "5px"
            },
            id={"type": "thumb", "index": i}
        )
        for i in thumb_indexes
    ]

    return main_img_src, label_text, thumbnails, current_index


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
