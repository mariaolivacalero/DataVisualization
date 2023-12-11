# Definir el layout
app.layout = html.Div(children=[
    # Sección superior con logo y título
    html.Div([
        html.Img(src=f'data:image/png;base64,{encoded_logo}', style={'height': '100px'}),
        html.H1("My year on Spotify", style={'text-align': 'center'}),
    ], style={'text-align': 'center', 'padding': '20px'}),

    # Sección izquierda con gráficas
    html.Div([
        graphyears,
        year,
        dcc.Graph(
            figure = px.bar(most_heard_30, 
             x='mins_played', 
             y='artistName', 
             orientation='h',  # Set orientation to horizontal for a bar plot
             title='Top 30 Artists Heard',
             labels={'artistName': 'Artist Name', 'mins_played': 'Minutes Played'},
             color='mins_played',  # Use 'mins_played' as the color variable
             color_continuous_scale='greens')),  # Set your desired color sequence) 
    ], style={'flex': '1', 'padding': '5px', 'float': 'left'}),

    # Sección derecha con dropdown y tabla
    html.Div([
        # Dropdown para seleccionar la playlist
        dcc.Dropdown(
            id='playlist-dropdown',
            options=playlist_options,
            value=playlist_options[0]['value'],  # Valor inicial
            multi=False,
            style={ 'margin-bottom': '20px'}
        ),
        
        # Tabla en Dash
        dash_table.DataTable(
            id='tabla',
            columns=[],
            data=[],
style_table={
                'height': '400px',
                'overflowY': 'auto',
                'backgroundColor': 'rgb(30 215 96)',  # Change background color to light green
            },
            style_header={
                'backgroundColor': 'rgb(30 215 96)',
                'color': 'white',
                'fontWeight': 'bold',
            },
            style_cell={
                'backgroundColor': 'rgb(152,251,152)',
                'color': 'black',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(144,238,144)',
                },
            ],
            virtualization=True,  # Enable virtualization for large datasets
        ), 
        html.Div([
        dcc.Graph(figure=scatter_loudness_energy),
        dcc.Graph(figure=scatter_valence_danceability),
    ], style={'display': 'flex'}),

    html.Div([
        dcc.Graph(figure=scatter_valence_energy),
        dcc.Graph(figure=scatter_loudness_valence),
    ], style={'display': 'flex'}),

    html.Div([
        dcc.Graph(figure=scatter_loudness_liveness),
        dcc.Graph(figure=scatter_danceability_tempo),
    ], style={'display': 'flex'})
    ], style={'flex': '1', 'padding': '20px', 'float': 'right', 'width': '40%'}),
])