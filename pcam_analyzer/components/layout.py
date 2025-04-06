import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from config import DEFAULT_MODEL_JSON, DEFAULT_MODEL_WEIGHTS
from models.model_loader import load_model
from utils.cam_utils import get_last_conv_layer
import tensorflow as tf

loaded_model = load_model()

dark_theme = {
    "backgroundColor": "#343a40",
    "color": "#ffffff",
    "minHeight": "100vh",
    "width": "100%",
    "margin": "0",
    "padding": "0",
}

def serve_layout():
    return html.Div(
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

                            html.H6("Loaded model: "),
                            html.Label(f"{DEFAULT_MODEL_JSON}"),
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
                            html.Label(f"{DEFAULT_MODEL_WEIGHTS}"),

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
                                                   style={"width": "275px"}),
                                        #className="mx-auto text-center",
                                        width=2,
                                    )
                                ]
                            ),

                            html.Img(id="model-visual2", src="assets/few_shot_confusion_matrix.png", style={"maxWidth": "350px", "paddingTop": "35px"}),

                            ],
                            width=2,
                            style={"padding": "20px"}
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
                                                    {"label": layer.name, "value": layer.name} for layer in loaded_model.layers if isinstance(layer, tf.keras.layers.Conv2D)
                                                ],
                                                value=get_last_conv_layer(loaded_model),  # Default: Last Conv Layer
                                                #clearable=False,
                                                style={"width": "10%", "margin-bottom": "10px"},
                                                className="mx-auto text-center",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Img(id="gradcam-image",
                                                                     style={"maxWidth": "300px", "borderRadius": "0px", "paddingTop": "0px"}),

                                                        width="auto", className="text-center"
                                                    ),
                                                    dbc.Col(
                                                        html.Img(id="xgrad-image",
                                                                 style={"maxWidth": "300px", "borderRadius": "0px", "paddingTop": "0px"}),
                                                        width="auto", className="text-center"
                                                    ),
                                                    dbc.Col(
                                                        children=[
                                                            html.Img(id="output-image",
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


                                                    dbc.Col(
                                                        html.Img(id="xrai-image",
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