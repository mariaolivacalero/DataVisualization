# My Streaming analysis

Toda la lógica que se ha utilizado para crear la APP se encuentra dentro de la carpeta Spotify.

 Spotify/ 
    app.py  código del Dash
    MyData/ datos históricos descargados de Spotify de los que se ha partido para el análisis
    analysis.ipynb analisis previo de los datos para la aplicacion
    cache.py busca los tokens de acceso a Spotify en el cache
    config.py contiene los credenciales de acceso a la API de Spotify for developers
    history.py contiene las funciones que llaman a los endpoints de Spotify para recoger los datos de las canciones
    data.py contiene la lógica del flujo de recogida y limpieza de datos para llegar al streaming.csv del que parte app.py
