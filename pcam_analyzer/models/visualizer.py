import visualkeras
from keras.utils import plot_model

def generate_model_visualization(model):
    visualkeras.layered_view(model, legend=True, to_file="assets/model_architecture.png")
    plot_model(model, to_file="assets/model_plot.png", show_shapes=True, show_layer_names=True)
