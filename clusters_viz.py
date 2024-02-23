from dash import Dash, dcc, html, Input, Output, no_update, State, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd



# Paths to your datasets organized by page
dataset_paths = {
    'p1': {
        'pca': 'data/pca_data_3D.csv',
        'umap': 'data/umap_data_3D.csv',
        'umap_pca': 'data/data_umap.csv',
        'tsne': 'data/tsne_data_3D.csv',
        'lda': 'data/lda_data_3D.csv',
    },
    'p2': {
        'pca': 'data/pca_data_3D_ts.csv',
        'umap': '/data/umap_data_3D_ts.csv',
        'umap_pca': '',
        'tsne': 'data/tsne_data_3D_ts.csv',
        'lda': 'data/lda_data_3D_ts.csv',
    },
    'p3': {
        'pca': 'data/pca_data_3D_mps.csv',
        'umap': 'data/umap_data_3D_mps.csv',
        'umap_pca': '',
        'tsne': 'data/tsne_data_3D_mps.csv',
        'lda': 'data/lda_data_3D_mps.csv',
    },
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def layout_page(page_id):
    # This function now accepts a page_id to differentiate between pages
    title_map = {
        'p1': "Computed Features",
        'p2': "Pitch Contour",
        'p3': "MPS",
    }
    return html.Div([
        dbc.Row(html.H4(title_map[page_id], className="mb-4 mt-4")),
        generate_page_content(page_id)  # Pass the page_id to the content generator
    ])

def create_homepage():
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(html.Img(src="assets/berkeley.png", height="100px"), width=12, className="text-center"),
            ], className="mb-5 mt-5"),
            dbc.Row([
                dbc.Col(html.H1("Recordings Clusters Visualization Dashboard", className="text-center"), width=12),
            ], className="mb-5"),
            dbc.Row([
                dbc.Col(html.Img(src="assets/bird.jpeg", style={"max-width": "100%", "height": "auto"}), md=4),
                dbc.Col(html.Img(src="assets/spec.jpg", style={"max-width": "100%", "height": "auto"}), md=4),
                dbc.Col(dcc.Markdown("""
                    **Welcome to the Clusters Visualization Dashboard!**

                    This dashboard provides insights into the acoustic properties of bird calls, featuring advanced visualization techniques such as PCA, UMAP, TSNE, and LDA.

                    Explore the tabs to dive into different aspects of the data.

                    ---
                    **Features:**
                    - Computed Features: Explore basic acoustic features.
                    - Pitch Contour: Analyze the pitch dynamics over time.
                    - MPS: Delve into the modulation power spectrum.

                    **Get Started:** Use the navigation bar to select a feature and begin your exploration.
                    """, className="text-justify"), md=4),
            ]),
        ], fluid=True, className="mt-5"),
    ])


def generate_page_content(page_id):
    # Use page_id to dynamically update the dataset selector based on the page
    return dbc.Row([
        dbc.Col([
            html.H5("Select Dataset:"),
            dbc.Select(
                id={
                    'type': 'dataset-selector', 
                    'index': page_id  # Use dictionary id to differentiate between pages
                },
                options=[
                    {'label': 'PCA', 'value': 'pca'},
                    {'label': 'UMAP', 'value': 'umap'},
                    {'label': 'UMAP with PCA features', 'value': 'umap_pca'},
                    {'label': 'T-SNE', 'value': 'tsne'},
                    {'label': 'LDA', 'value': 'lda'}
                ],
                value='pca',  # Default value
            ),
        ], width=3),
        dbc.Col([
            dcc.Graph(
                id={
                    'type': 'graph-basic', 
                    'index': page_id  # Use dictionary id for the graph as well
                }, 
                clear_on_unhover=True, 
                style={'height': '800px'}
            ),
            dcc.Tooltip(id={
                'type': 'graph-tooltip', 
                'index': page_id  # And for the tooltip
            }),
        ], width=9),
    ])

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Computed Features", href="/computed-features"),
                    dbc.DropdownMenuItem("Pitch Contour", href="/pitch-contour"),
                    dbc.DropdownMenuItem("MPS", href="/mps"),
                ],
                nav=True,
                in_navbar=True,
                label="Explore Clusters",
            ),
        ],
        brand="Clusters Visualization",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    html.Div(id='page-content'),  
], fluid=True)

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/computed-features':
        return layout_page('p1')
    elif pathname == '/pitch-contour':
        return layout_page('p2')
    elif pathname == '/mps':
        return layout_page('p3')
    else:
        return create_homepage()  # Return the homepage layout for the root URL

# Callback to update graph based on dataset selection
@app.callback(
    Output({'type': 'graph-basic', 'index': MATCH}, "figure"),
    [Input({'type': 'dataset-selector', 'index': MATCH}, "value"),
     Input('url', 'pathname')]
)
def update_graph(selected_dataset, pathname):
    page_id = 'p1'  # Default to p1, adjust based on pathname
    if pathname == '/computed-features':
        page_id = 'p1'
    elif pathname == '/pitch-contour':
        page_id = 'p2'
    elif pathname == '/mps':
        page_id = 'p3'

    # Use the correct dataset path based on the page
    dataset_path = dataset_paths[page_id][selected_dataset]
    df = pd.read_csv(dataset_path)

    # Create figure based on the selected dataset
    fig = go.Figure(data=[
        go.Scatter3d(
            x=df['Dim1'], 
            y=df['Dim2'], 
            z=df['Dim3'],
            mode='markers',
            marker=dict(
                color=df['color'],          
                opacity=1,                
                line=dict(
                    color='black',          
                    width=2                 
                )
            )
        )
    ])

    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    return fig

@app.callback(
    Output({'type': 'graph-tooltip', 'index': MATCH}, "show"),
    Output({'type': 'graph-tooltip', 'index': MATCH}, "bbox"),
    Output({'type': 'graph-tooltip', 'index': MATCH}, "children"),
    [Input({'type': 'graph-basic', 'index': MATCH}, "hoverData"),
     State({'type': 'dataset-selector', 'index': MATCH}, "value"),
     State('url', 'pathname')]  
)
def display_hover(hoverData, selected_dataset, pathname):
    if hoverData is None:
        return False, no_update, no_update

    # Extract the page_id from the pathname
    page_id = 'p1'  # Default, adjust based on your actual URL structure
    if pathname.endswith('computed-features'):
        page_id = 'p1'
    elif pathname.endswith('pitch-contour'):
        page_id = 'p2'
    elif pathname.endswith('mps'):
        page_id = 'p3'

    # Use the correct dataset path based on the page
    dataset_path = dataset_paths[page_id][selected_dataset]
    df = pd.read_csv(dataset_path)

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src = df_row['image_url']

    children = html.Div([
        html.P([html.Span("Bird: ", style={'font-weight': 'bold'}), f"{df_row['bird']}"], style={'font-size': '15px', 'margin': '7px', 'line-height': '1.2'}),
        html.P([html.Span("Morph: ", style={'font-weight': 'bold'}), f"{df_row['morph']}"], style={'font-size': '15px', 'margin': '7px', 'line-height': '1.2'}),
        html.P([html.Span("Population: ", style={'font-weight': 'bold'}), f"{df_row['pop_code']}"], style={'font-size': '15px', 'margin': '7px', 'line-height': '1.2'}),
        html.P([html.Span("Syllable: ", style={'font-weight': 'bold'}), f"{df_row['syllable']}"], style={'font-size': '15px', 'margin': '7px', 'line-height': '1.2'}),
        html.Img(src=img_src, style={"width": "100%"}),
    ], style={'width': '200px', 'white-space': 'normal'})

    return True, bbox, children

if __name__ == "__main__":
    app.run_server(debug=True)
