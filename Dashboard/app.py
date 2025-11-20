import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px

# 1. Load the Dataset
# Ensure this file is in the same directory
#df = pd.read_csv('nyc_collision_data_cleaned.csv')
df = pd.read_parquet('crashes.parquet')

# Pre-processing for Dropdowns (Extract unique values for filters)
# You may need to adjust column names based on your specific cleaning
boroughs = df['BOROUGH'].unique().tolist()
# --- FIX START ---
# Your file has 'CRASH_DATETIME', so we create 'CRASH_DATE' from it.
# We convert it to datetime objects to be safe.
df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
df['CRASH_DATE'] = df['CRASH_DATETIME'].dt.date

# Now we can safely extract the years
# We drop NaNs (Not a Number) to avoid errors in the dropdown
years = sorted(df['CRASH_DATETIME'].dt.year.dropna().astype(int).unique())
# --- FIX END ---
# 2. Initialize the App
app = dash.Dash(__name__)
server = app.server # Needed for Vercel/Heroku deployment later 

# 3. Define the Layout
app.layout = html.Div([
    html.H1("NYC Motor Vehicle Collisions Report", style={'textAlign': 'center'}),

    # --- CONTROL PANEL ---
    html.Div([
        html.H3("Filters"),
        
        # Requirement: Multiple dropdown filters [cite: 63]
        html.Label("Select Borough:"),
        dcc.Dropdown(
            id='filter-borough',
            options=[{'label': b, 'value': b} for b in boroughs],
            multi=True, # Allow selecting multiple boroughs
            placeholder="Select Borough(s)..."
        ),

        html.Label("Select Year:"),
        dcc.Dropdown(
            id='filter-year',
            options=[{'label': y, 'value': y} for y in years],
            multi=True,
            placeholder="Select Year(s)..."
        ),
        
        # Add the other required dropdowns here: Vehicle Type, Contributing Factor, Injury Type [cite: 63]
        
        # Requirement: Search Mode 
        html.Label("Search (e.g., 'Pedestrian'):"),
        dcc.Input(id='filter-search', type='text', placeholder='Type query...', style={'width': '100%'}),

        html.Br(), html.Br(),

        # Requirement: Central "Generate Report" Button 
        html.Button('Generate Report', id='btn-generate', n_clicks=0, style={'fontSize': '16px', 'padding': '10px'})
    
    ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f9f9f9'}),

    # --- VISUALIZATION PANEL ---
    html.Div([
        html.H3("Dashboard Analytics"),
        
        # Placeholders for your charts [cite: 66]
        dcc.Graph(id='chart-1'),
        dcc.Graph(id='chart-2')
        
    ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px'})
])

# 4. Define the Callback
@app.callback(
    [Output('chart-1', 'figure'),
     Output('chart-2', 'figure')],
    [Input('btn-generate', 'n_clicks')],
    [State('filter-borough', 'value'),
     State('filter-year', 'value'),
     State('filter-search', 'value')]
)
def update_report(n_clicks, selected_boroughs, selected_years, search_query):
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    # --- DEBUG PRINT 1 ---
    print(f"\n--- GENERATING REPORT (Click {n_clicks}) ---")
    print(f"Total rows in dataset: {len(df)}")
    print(f"Filters -> Boroughs: {selected_boroughs}, Years: {selected_years}, Search: '{search_query}'")

    dff = df.copy()

    # 1. Filter by Borough
    if selected_boroughs:
        dff = dff[dff['BOROUGH'].isin(selected_boroughs)]
        print(f"Rows after Borough filter: {len(dff)}")

    # 2. Filter by Year
    if selected_years:
        # Ensure column is datetime
        if not pd.api.types.is_datetime64_any_dtype(dff['CRASH_DATETIME']):
             dff['CRASH_DATETIME'] = pd.to_datetime(dff['CRASH_DATETIME'], errors='coerce')
        
        dff = dff[dff['CRASH_DATETIME'].dt.year.isin(selected_years)]
        print(f"Rows after Year filter: {len(dff)}")

    # 3. Filter by Search Query
    if search_query:
        # EXPANDED SEARCH: Check more columns for better results
        # We verify if the column exists before searching to prevent errors
        search_cols = [c for c in ['BOROUGH', 'ON STREET NAME', 'CONTRIBUTING FACTOR VEHICLE 1', 'VEHICLE TYPE CODE 1'] if c in dff.columns]
        
        mask = dff[search_cols].apply(lambda x: x.astype(str).str.contains(search_query, case=False, na=False)).any(axis=1)
        dff = dff[mask]
        print(f"Rows after Search filter: {len(dff)}")

    # --- DEBUG PRINT FINAL ---
    print(f"FINAL rows to plot: {len(dff)}")

    if dff.empty:
        print("!!! RESULT IS EMPTY - RETURNING BLANK CHARTS !!!")
        # Return a text annotation saying "No Data"
        fig_empty = px.bar(title="No data found matching these filters.")
        return fig_empty, fig_empty

    # 4. Generate Visualizations
    # Bar Chart: Crash Count by Borough
    dff_counts = dff['BOROUGH'].value_counts().reset_index()
    dff_counts.columns = ['BOROUGH', 'Count']
    fig1 = px.bar(dff_counts, x='BOROUGH', y='Count', title="Crashes by Borough")

    # Map: Crash Locations
    # Limit to 1000 points for speed
    fig2 = px.scatter_mapbox(dff.head(1000), lat="LATITUDE", lon="LONGITUDE", zoom=10, title="Crash Locations (Sample)")
    fig2.update_layout(mapbox_style="open-street-map")

    return fig1, fig2

# 5. Run the App
if __name__ == '__main__':
    app.run(debug=True)