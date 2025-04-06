import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
import utils.cam_utils as cam_utils
from utils.preprocessing import stain_standardization
from utils.image_utils import img_to_base64, load_images
from models.model_loader import load_model


loaded_model = load_model()
image_list = load_images()
num_images = len(image_list)

def register_callbacks(app):
    @app.callback(
        [Output("output-image", "src"),
         Output("gradcam-image", "src"),
         Output("hircam-image", "src"),
         Output("xrai-image", "src"),
         Output("xgrad-image", "src"),
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
    def update_image(startup, prev_clicks, next_clicks, flag_btn, thumb_clicks, opacity, selected_layer, current_index,
                     flagged_samples, visited_images):
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
                flagged_samples.append(filename)  # Save filename of flagged image

        filename = image_list[current_index][1]
        if filename not in visited_images:
            visited_images.append(filename)  # Mark as visited

        img_array, filename = image_list[current_index]

        main_img_src = img_to_base64(img_array, size=(96, 96), title="Original")
        orig_img = img_array.copy()

        img_array = stain_standardization(img_array)

        # Compute Grad-CAM Heatmap
        heatmap_gradcam = cam_utils.compute_grad_cam(loaded_model, img_array, layer_name=selected_layer)
        gradcam_overlay = cam_utils.overlay_heatmap(orig_img, heatmap_gradcam, opacity)

        heatmap = cam_utils.compute_hires_cam(loaded_model, img_array, layer_name=selected_layer)
        hircam_overlay = cam_utils.overlay_heatmap(orig_img, heatmap, opacity)

        # heatmap = compute_score_cam(loaded_model, img_array, last_conv_layer_name=selected_layer)
        overlayed_scorecam = gradcam_overlay  # overlay_heatmap(img_array, heatmap, opacity)
        # compute_xrai_heatmap(loaded_model, img_array)

        xgrad_heatmap = cam_utils.compute_xgrad_cam(loaded_model, img_array, last_conv_layer_name=selected_layer)
        # xrai_heatmap = compute_xrai_heatmap(loaded_model, img_array)

        #  Convert to Overlay Heatmap
        overlayed_xgrad = cam_utils.overlay_heatmap(orig_img, xgrad_heatmap, alpha=opacity)
        # Extract Confidence Score from Filename
        # confidence = 100 - int(filename.split("_")[0])  # Extract confidence
        confidence = 0
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
        return (main_img_src, img_to_base64(gradcam_overlay, size=(96, 96), title="GradCAM"),
                img_to_base64(hircam_overlay, size=(96, 96), title="HiResCAM"),
                img_to_base64(overlayed_scorecam, title="ScoreCAM"), img_to_base64(overlayed_xgrad, title="xGRAD-CAM"),
                label, confidence_text, thumbnails,
                current_index, flagged_samples, visited_images, f"Export {flagged_count} Flagged Images")

