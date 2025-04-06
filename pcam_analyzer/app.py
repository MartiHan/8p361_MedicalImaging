import dash
import dash_bootstrap_components as dbc
from components.layout import serve_layout
from callbacks.main_callbacks import register_callbacks

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
app.title = "PCam Analyzer"
app.layout = serve_layout()
app.config.suppress_callback_exceptions = True

register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True)