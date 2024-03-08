from dash import Dash, dcc, html, Input, Output, no_update, State, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
import numpy as np



dataset_paths = {
    'p1': {
        'pca': 'data/pca_data_3D.csv',
        'umap': 'data/umap_data_3D.csv',
        'umap_pca': 'data/umap_dataPCA_3D.csv',
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

    'p4': {
        'pca': 'data/pca_data_3D_spectro.csv',
        'umap': 'data/umap_data_3D_spectro.csv',
        'umap_pca': '',
        'tsne': 'data/tsne_data_3D_spectro.csv',
        'lda': 'data/lda_data_3D_spectro.csv',
    },

    'p5': {
        'pca': 'data/pca_data_3D_vae.csv',
        'umap': 'data/umap_data_3D_vae.csv',
        'umap_pca': '',
        'tsne': 'data/tsne_data_3D_vae.csv',
        'lda': 'data/lda_data_3D_vae.csv',
    },
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def layout_page(page_id):
    title_map = {
        'p1': "Computed Features",
        'p2': "Pitch Contour Features",
        'p3': "MPS Features",
        'p4': "Spectro Features",
        'p5': 'Latent features'
    }
    return html.Div([
        dbc.Row(html.H4(title_map[page_id], className="mb-4 mt-4")),
        generate_page_content(page_id)  
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
                    - Traditional Features: Explore basic acoustic features.
                    - Pitch Contour: Analyze the pitch dynamics over time.
                    - MPS: Delve into the modulation power spectrum.
                    - Spectro: Use the array of the spectrogram as features. 
                    - VAE: Explore the latent features from a tunned variational autoencoder

                    **Get Started:** Use the navigation bar to select a feature and begin your exploration.
                    """, className="text-justify"), md=4),
            ]),
        ], fluid=True, className="mt-5"),
    ])


def generate_page_content(page_id):
    # Column for selecting the dataset
    dataset_selector_col = dbc.Col([
        html.H5("Select Dataset:"),
        dbc.Select(
            id={
                'type': 'dataset-selector',
                'index': page_id
            },
            options=[
                {'label': 'PCA', 'value': 'pca'},
                {'label': 'UMAP', 'value': 'umap'},
                {'label': 'UMAP with PCA features', 'value': 'umap_pca'},
                {'label': 'T-SNE', 'value': 'tsne'},
                {'label': 'LDA', 'value': 'lda'}
            ],
            value='pca',
        ),
        html.Br(),
        html.Div([
            html.H6("Filter by: Bird"),
            dbc.Select(
                id={'type': 'filter-bird', 'index': page_id},
                options=[],  # Options will be dynamically generated
                value='all',  # Set 'All' as default value
                placeholder="Select a Bird",
            ),
        ]),
        html.Br(),
        html.Div([
            html.H6("Filter by: Population"),
            dbc.Select(
                id={'type': 'filter-population', 'index': page_id},
                options=[],  # Options will be dynamically generated
                value='all',  # Set 'All' as default value
                placeholder="Select a Population",
            ),
        ]),
        html.Br(),
        html.Div([
            html.H6("Filter by: Morph"),
            dbc.Select(
                id={'type': 'filter-morph', 'index': page_id},
                options=[],  # Options will be dynamically generated
                value='all',  # Set 'All' as default value
                placeholder="Select a Morph",
            ),
        ]),
    ], width=3)

    # Column for the graph
    graph_col = dbc.Col([
        dcc.Graph(
            id={
                'type': 'graph-basic',
                'index': page_id
            },
            clear_on_unhover=True,
            style={'height': '800px'}
        ),
        dcc.Tooltip(id={
            'type': 'graph-tooltip',
            'index': page_id
        }),
    ], width=9)

    # Combine columns into a single row, with filtering options now vertically aligned
    return dbc.Row([dataset_selector_col, graph_col])



app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Computed Features", href="/computed-features"),
                    dbc.DropdownMenuItem("Pitch Contour Features", href="/pitch-contour"),
                    dbc.DropdownMenuItem("MPS Features", href="/mps"),
                    dbc.DropdownMenuItem("Spectro Features", href="/spec"),
                    dbc.DropdownMenuItem("VAE Latent Features", href="/vae"),

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
    elif pathname == '/spec':
        return layout_page('p4')
    elif pathname == '/vae':
        return layout_page('p5')
    else:
        return create_homepage()  # Return the homepage layout for the root URL
    

@app.callback(
    [Output({'type': 'filter-bird', 'index': MATCH}, 'options'),
     Output({'type': 'filter-morph', 'index': MATCH}, 'options'),
     Output({'type': 'filter-population', 'index': MATCH}, 'options')],
    [Input({'type': 'dataset-selector', 'index': MATCH}, 'value')],
    [State('url', 'pathname')]
)
def update_filter_options(selected_dataset, pathname):
    page_id = 'p1' 
    if pathname == '/computed-features':
        page_id = 'p1'
    elif pathname == '/pitch-contour':
        page_id = 'p2'
    elif pathname == '/mps':
        page_id = 'p3'
    elif pathname == '/spec':
        page_id = 'p4'
    elif pathname == '/vae':
        page_id = 'p5'

    dataset_path = dataset_paths[page_id][selected_dataset]
    df = pd.read_csv(dataset_path)
    bird_options = [{'label': 'All', 'value': 'all'}] + [{'label': bird, 'value': bird} for bird in df['bird'].unique()]
    morph_options = [{'label': 'All', 'value': 'all'}] + [{'label': morph, 'value': morph} for morph in df['morph'].unique()]
    pop_options = [{'label': 'All', 'value': 'all'}] + [{'label': pop, 'value': pop} for pop in df['pop_code'].unique()]
    
    return bird_options, morph_options, pop_options


def generate_syllable_colormap(df, column_name):
    # Assuming all syllables are now integers in the column
    unique_syllables = sorted(df[column_name].dropna().unique())
    
    # Generate a continuous Viridis color scale
    num_syllables = len(unique_syllables)
    viridis_scale = px.colors.sequential.Viridis
    colors = [viridis_scale[int(i * (len(viridis_scale)-1) / (num_syllables-1))] for i in range(num_syllables)]
    
    # Map each syllable to a color
    syllable_colors = {syllable: colors[i] for i, syllable in enumerate(unique_syllables)}
    
    return syllable_colors



def generate_pop_morph_colormap(df, syllable_column='syllable'):
    unique_syllables = df[syllable_column].dropna().unique()
    color_scale = px.colors.qualitative.Vivid  
    syllable_colors = {syllable: color_scale[i % len(color_scale)] for i, syllable in enumerate(unique_syllables)}
    return syllable_colors

@app.callback(
    Output({'type': 'graph-basic', 'index': MATCH}, "figure"),
    [Input({'type': 'dataset-selector', 'index': MATCH}, "value"),
     Input({'type': 'filter-bird', 'index': MATCH}, "value"),
     Input({'type': 'filter-morph', 'index': MATCH}, "value"),
     Input({'type': 'filter-population', 'index': MATCH}, "value"),
     Input('url', 'pathname')]
)
def update_graph(selected_dataset, selected_bird, selected_morph, selected_population, pathname):
    page_id = 'p1' 
    if pathname == '/computed-features':
        page_id = 'p1'
    elif pathname == '/pitch-contour':
        page_id = 'p2'
    elif pathname == '/mps':
        page_id = 'p3'
    elif pathname == '/spec':
        page_id = 'p4'
    elif pathname == '/vae':
        page_id = 'p5'

    dataset_path = dataset_paths.get(page_id, {}).get(selected_dataset, '')
    if not dataset_path:
        return go.Figure()  # Return an empty figure if the dataset path isn't found
    df = pd.read_csv(dataset_path)

    df['syllable'] = df['syllable'].apply(lambda x: int(x.replace('S', '')) if isinstance(x, str) else x)
    

    # Initialize selection column
    df['is_selected'] = True

    # Apply selection filters
    if selected_bird != 'all':
        df['is_selected'] &= (df['bird'] == selected_bird)
    if selected_morph != 'all':
        df['is_selected'] &= (df['morph'] == selected_morph)
    if selected_population != 'all':
        df['is_selected'] &= (df['pop_code'] == selected_population)

    # Determine the coloring logic based on the selection
    if selected_bird != 'all':
        color_by = 'syllable'
        color_map = generate_syllable_colormap(df[df['is_selected']], 'syllable')
    elif selected_morph != 'all':
        color_by = 'pop_code'
        color_map = generate_pop_morph_colormap(df[df['is_selected']], 'pop_code')
    elif selected_population != 'all':
        color_by = 'bird'
        color_map = generate_pop_morph_colormap(df[df['is_selected']], 'bird')
    else:
        color_by = 'morph'
        color_map = generate_pop_morph_colormap(df, 'morph')

    # Apply the color mapping to selected data points
    df['color'] = pd.Series(dtype='object')
    df.loc[df['is_selected'], 'color'] = df[df['is_selected']][color_by].map(color_map).fillna('lightgrey')
    df.loc[~df['is_selected'], 'color'] = 'lightgrey'

    # Plotting setup
    fig = go.Figure()

    # Add selected points
    for label, group in df[df['is_selected']].groupby(color_by):
        fig.add_trace(go.Scatter3d(
            x=group['Dim1'],
            y=group['Dim2'],
            z=group['Dim3'],
            mode='markers',
            marker=dict(size=5, opacity=1, line=dict(color='black', width=2), color=group['color']),
            name=str(label)  # Set the legend name to the label of the group
        ))

    # Add unselected points
    if df[~df['is_selected']].any().any():
        fig.add_trace(go.Scatter3d(
            x=df[~df['is_selected']]['Dim1'],
            y=df[~df['is_selected']]['Dim2'],
            z=df[~df['is_selected']]['Dim3'],
            mode='markers',
            hoverinfo='none',
            marker=dict(size=3, color='lightgrey', opacity=0.25),
            name="Unselected"
        ))

    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        hovermode='closest',
        scene=dict(
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False)
        ),
        legend_title_text=color_by.capitalize()  # Update legend title based on what we are coloring by
    )
    return fig



@app.callback(
    [Output({'type': 'graph-tooltip', 'index': MATCH}, "show"),
     Output({'type': 'graph-tooltip', 'index': MATCH}, "bbox"),
     Output({'type': 'graph-tooltip', 'index': MATCH}, "children")],
    [Input({'type': 'graph-basic', 'index': MATCH}, "hoverData")],
    [State({'type': 'dataset-selector', 'index': MATCH}, "value"),
     State({'type': 'filter-bird', 'index': MATCH}, "value"),
     State({'type': 'filter-morph', 'index': MATCH}, "value"),
     State({'type': 'filter-population', 'index': MATCH}, "value"),
     State('url', 'pathname')]
)
def display_hover(hoverData, selected_dataset, selected_bird, selected_morph, selected_population, pathname):
    if hoverData is None:
        return False, no_update, no_update

    # Extract the page_id from the pathname
    page_id = 'p1'  
    if pathname.endswith('computed-features'):
        page_id = 'p1'
    elif pathname.endswith('pitch-contour'):
        page_id = 'p2'
    elif pathname.endswith('mps'):
        page_id = 'p3'
    elif pathname.endswith('spec'):
        page_id = 'p4'
    elif pathname.endswith('vae'):
        page_id = 'p5'

    dataset_path = dataset_paths[page_id][selected_dataset]
    df = pd.read_csv(dataset_path)
    df['is_selected'] = True

    if selected_bird != 'all':
        df['is_selected'] &= (df['bird'] == selected_bird)
    if selected_morph != 'all':
        df['is_selected'] &= (df['morph'] == selected_morph)
    if selected_population != 'all':
        df['is_selected'] &= (df['pop_code'] == selected_population)
    
  
    selected_count = df[df['is_selected']].shape[0]
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    point_index = pt["pointNumber"]

    if point_index >= selected_count:
        return False, no_update, no_update

    df_row = df[df['is_selected']].iloc[point_index]

    # Check if the hovered point is selected
    if not df_row['is_selected']:
        # If the point is not selected, do not show the tooltip
        return False, no_update, no_update

    # If the point is selected, proceed to show the tooltip as before
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
