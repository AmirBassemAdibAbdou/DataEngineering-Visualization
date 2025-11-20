import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px

# 1. Load the Dataset
# We use PyArrow for the whole dataframe to save memory
df = pd.read_parquet('crashes.parquet', engine='pyarrow', dtype_backend='pyarrow')

# --- PRE-PROCESSING ---

# A. Handle Dates
# We convert the date column to standard numpy 'datetime64[ns]' to avoid Windows TimeZone errors
df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce').astype('datetime64[ns]')
df['DATE_STR'] = df['CRASH_DATETIME'].dt.strftime('%Y-%m-%d')

# B. Extract Unique Values for Dropdowns
boroughs = sorted(df['BOROUGH'].dropna().unique().tolist())
years = sorted(df['CRASH_DATETIME'].dt.year.dropna().astype(int).unique())

# Vehicle Types (Limit to top 200 for performance)
vehicle_types = df['VEHICLE TYPE CODE 1'].dropna().astype(str).unique().tolist()
vehicle_types = sorted(vehicle_types)[:200] 

# Contributing Factors
factors = df['CONTRIBUTING FACTOR VEHICLE 1'].dropna().astype(str).unique().tolist()
factors = sorted(factors)

# Collision Severity Options
severity_options = ['Fatality', 'Injury', 'Property Damage Only']

# 2. Initialize the App
app = dash.Dash(__name__)
server = app.server 

# 3. Define the Layout
app.layout = html.Div([
    html.H1("NYC Motor Vehicle Collisions Report", style={'textAlign': 'center', 'fontFamily': 'Arial'}),

    html.Div([
        # --- CONTROL PANEL (Left Side) ---
        html.Div([
            html.H3("Filters", style={'borderBottom': '2px solid #007bff', 'paddingBottom': '10px'}),
            
            # 1. Search Box
            html.Label("Global Search:", style={'fontWeight': 'bold'}),
            dcc.Input(
                id='filter-search', 
                type='text', 
                placeholder='e.g. "Brooklyn 2022"', 
                style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}
            ),

            # 2. Borough Filter
            html.Label("Borough:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='filter-borough',
                options=[{'label': b, 'value': b} for b in boroughs],
                multi=True,
                placeholder="Select Borough(s)...",
                style={'marginBottom': '10px'}
            ),

            # 3. Year Filter
            html.Label("Year:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='filter-year',
                options=[{'label': y, 'value': y} for y in years],
                multi=True,
                placeholder="Select Year(s)...",
                style={'marginBottom': '10px'}
            ),

            # 4. Vehicle Type Filter
            html.Label("Vehicle Type:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='filter-vehicle',
                options=[{'label': v, 'value': v} for v in vehicle_types],
                multi=True,
                placeholder="e.g. Sedan, Taxi...",
                style={'marginBottom': '10px'}
            ),

            # 5. Contributing Factor Filter
            html.Label("Contributing Factor:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='filter-factor',
                options=[{'label': f, 'value': f} for f in factors],
                multi=True,
                placeholder="e.g. Alcohol, Speeding...",
                style={'marginBottom': '10px'}
            ),

            # 6. Severity Filter
            html.Label("Collision Severity:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='filter-severity',
                options=[{'label': s, 'value': s} for s in severity_options],
                multi=True,
                placeholder="Select Severity...",
                style={'marginBottom': '20px'}
            ),

            # Generate Button
            html.Button(
                'Generate Report', 
                id='btn-generate', 
                n_clicks=0, 
                style={
                    'width': '100%', 'backgroundColor': '#007bff', 'color': 'white', 
                    'border': 'none', 'padding': '12px', 'fontSize': '16px', 
                    'cursor': 'pointer', 'borderRadius': '5px'
                }
            )
        
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f1f1f1', 'borderRadius': '10px'}),

        # --- VISUALIZATION PANEL (Right Side) ---
        html.Div([
            html.H3("Dashboard Analytics"),
            
            # Key Metrics Row
            html.Div(id='stats-container', style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),

            dcc.Graph(id='chart-1'),
            html.Br(),
            dcc.Graph(id='chart-2')
            
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'})
])

# 4. Define the Callback
@app.callback(
    [Output('chart-1', 'figure'),
     Output('chart-2', 'figure'),
     Output('stats-container', 'children')],
    [Input('btn-generate', 'n_clicks')],
    [State('filter-borough', 'value'),
     State('filter-year', 'value'),
     State('filter-vehicle', 'value'),
     State('filter-factor', 'value'),
     State('filter-severity', 'value'),
     State('filter-search', 'value')]
)
def update_report(n_clicks, sel_boroughs, sel_years, sel_vehicles, sel_factors, sel_severity, search_query):
    if n_clicks == 0:
        return dash.no_update, dash.no_update, dash.no_update

    print(f"--- GENERATING REPORT ---")
    dff = df.copy()

    # --- APPLY DROPDOWN FILTERS ---
    if sel_boroughs:
        dff = dff[dff['BOROUGH'].isin(sel_boroughs)]
    
    if sel_years:
        dff = dff[dff['CRASH_DATETIME'].dt.year.isin(sel_years)]

    if sel_vehicles:
        dff = dff[dff['VEHICLE TYPE CODE 1'].isin(sel_vehicles)]

    if sel_factors:
        dff = dff[dff['CONTRIBUTING FACTOR VEHICLE 1'].isin(sel_factors)]

    if sel_severity:
        conditions = []
        if 'Fatality' in sel_severity:
            conditions.append(dff['NUMBER OF PERSONS KILLED'] > 0)
        if 'Injury' in sel_severity:
            conditions.append(dff['NUMBER OF PERSONS INJURED'] > 0)
        if 'Property Damage Only' in sel_severity:
            conditions.append((dff['NUMBER OF PERSONS KILLED'] == 0) & (dff['NUMBER OF PERSONS INJURED'] == 0))
        
        if conditions:
            final_condition = conditions[0]
            for cond in conditions[1:]:
                final_condition = final_condition | cond
            dff = dff[final_condition]

    # --- APPLY SEARCH QUERY (FIXED FOR MEMORY) ---
    if search_query:
        terms = search_query.split()
        
        search_cols = [
            'BOROUGH', 
            'ON STREET NAME', 
            'CONTRIBUTING FACTOR VEHICLE 1', 
            'VEHICLE TYPE CODE 1',
            'DATE_STR'
        ]
        
        # Filter: ALL terms must appear in the row
        for term in terms:
            # Initialize a mask of all False (safe initialization)
            term_mask = None
            
            for col in search_cols:
                if col not in dff.columns:
                    continue
                
                try:
                    # MEMORY FIX: Process ONE column at a time.
                    # We access the column, convert to string, check contains, then discard the string object.
                    # This prevents allocating a massive matrix of strings for all columns at once.
                    col_series = dff[col].astype(str)
                    
                    # Check match
                    col_matches = col_series.str.contains(term, case=False, na=False)
                    
                    if term_mask is None:
                        term_mask = col_matches
                    else:
                        # Logical OR: If it matches in this column OR previous columns, keep the row
                        term_mask = term_mask | col_matches
                        
                except Exception as e:
                    print(f"Skipping column {col} due to error: {e}")
                    continue
            
            # Apply the mask for this specific search term
            if term_mask is not None:
                dff = dff[term_mask]

    if dff.empty:
        empty_fig = px.bar(title="No data matches your filters.")
        return empty_fig, empty_fig, html.Div("No Data Found")

    # --- GENERATE CHARTS ---
    
    if sel_boroughs and len(sel_boroughs) == 1:
        group_col = 'CONTRIBUTING FACTOR VEHICLE 1'
        title = f"Top Contributing Factors in {sel_boroughs[0]}"
    else:
        group_col = 'BOROUGH'
        title = "Crashes by Borough"

    dff_counts = dff[group_col].value_counts().reset_index().head(15)
    dff_counts.columns = [group_col, 'Count']
    fig1 = px.bar(dff_counts, x=group_col, y='Count', title=title, text_auto=True)
    fig1.update_layout(xaxis={'categoryorder':'total descending'})

    # Chart 2: Map
    # Limit to 1000 points to save browser memory
    map_data = dff.dropna(subset=['LATITUDE', 'LONGITUDE']).head(1000)
    fig2 = px.scatter_mapbox(
        map_data, 
        lat="LATITUDE", 
        lon="LONGITUDE", 
        zoom=10, 
        title=f"Crash Locations ({len(map_data)} displayed)",
        color_discrete_sequence=["red"]
    )
    fig2.update_layout(mapbox_style="open-street-map")

    # Stats
    total_crashes = len(dff)
    total_injured = dff['NUMBER OF PERSONS INJURED'].sum()
    total_killed = dff['NUMBER OF PERSONS KILLED'].sum()

    stats = [
        html.Div([html.H4("Total Crashes"), html.P(f"{total_crashes:,}")]),
        html.Div([html.H4("Total Injured"), html.P(f"{total_injured:,}")]),
        html.Div([html.H4("Total Killed"), html.P(f"{total_killed:,}")])
    ]

    return fig1, fig2, stats

if __name__ == '__main__':
    app.run(debug=True)