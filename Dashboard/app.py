import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px

# 1. Load the Dataset
print("Loading parquet file (this may take a moment)...")
df = pd.read_parquet('crashes.parquet', engine='pyarrow', dtype_backend='pyarrow')
print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

# Optimize dtypes to save memory
categorical_cols = ['BOROUGH', 'CONTRIBUTING FACTOR VEHICLE 1', 'VEHICLE TYPE CODE 1', 'PERSON_TYPE', 'PERSON_INJURY']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Pre-processing for Dropdowns (Extract unique values for filters)
boroughs = sorted(df['BOROUGH'].dropna().unique().tolist())

# Extract years - use CRASH YEAR if available, otherwise parse from CRASH_DATETIME
if 'CRASH YEAR' in df.columns:
    years = sorted(df['CRASH YEAR'].dropna().astype(int).unique())
else:
    # Fallback: parse from CRASH_DATETIME
    df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
    df['CRASH_DATE'] = df['CRASH_DATETIME'].dt.date
    years = sorted(df['CRASH_DATETIME'].dt.year.dropna().astype(int).unique())

# Vehicle Types (Limit to top 200 for performance)
vehicle_types = df['VEHICLE TYPE CODE 1'].dropna().astype(str).unique().tolist()
vehicle_types = sorted(vehicle_types)[:200]

# Contributing Factors
factors = df['CONTRIBUTING FACTOR VEHICLE 1'].dropna().astype(str).unique().tolist()
factors = sorted(factors)

# Collision Severity Options
severity_options = ['Fatality', 'Injury', 'Property Damage Only']

print(f"Data loaded successfully! {len(df):,} rows, {len(boroughs)} boroughs, years {min(years)}-{max(years)}")
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
            
            # Grid layout for multiple chart types
            html.Div([
                # Row 1: Bar Chart and Line Chart
                html.Div([
                    dcc.Graph(id='chart-bar', style={'display': 'inline-block', 'width': '48%'}),
                    dcc.Graph(id='chart-line', style={'display': 'inline-block', 'width': '48%'})
                ], style={'width': '100%'}),
                
                # Row 2: Heatmap and Pie Chart
                html.Div([
                    dcc.Graph(id='chart-heatmap', style={'display': 'inline-block', 'width': '48%'}),
                    dcc.Graph(id='chart-pie', style={'display': 'inline-block', 'width': '48%'})
                ], style={'width': '100%', 'marginTop': '20px'}),
                
                # Row 3: Map (full width)
                html.Div([
                    dcc.Graph(id='chart-map', style={'width': '100%'})
                ], style={'width': '100%', 'marginTop': '20px'})
            ])
            
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'})
    # --- VISUALIZATION PANEL ---
    html.Div([
        html.H3("Dashboard Analytics"),
        
        # Grid layout for multiple chart types
        html.Div([
            # Row 1: Bar Chart and Line Chart
            html.Div([
                dcc.Graph(id='chart-bar', style={'display': 'inline-block', 'width': '48%'}),
                dcc.Graph(id='chart-line', style={'display': 'inline-block', 'width': '48%'})
            ], style={'width': '100%'}),
            
            # Row 2: Heatmap and Pie Chart
            html.Div([
                dcc.Graph(id='chart-heatmap', style={'display': 'inline-block', 'width': '48%'}),
                dcc.Graph(id='chart-pie', style={'display': 'inline-block', 'width': '48%'})
            ], style={'width': '100%', 'marginTop': '20px'}),
            
            # Row 3: Map (full width)
            html.Div([
                dcc.Graph(id='chart-map', style={'width': '100%'})
            ], style={'width': '100%', 'marginTop': '20px'})
        ])
        
    ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px'})
])

# 4. Define the Callback - Returns all chart types
@app.callback(
    [Output('chart-bar', 'figure'),
     Output('chart-line', 'figure'),
     Output('chart-heatmap', 'figure'),
     Output('chart-pie', 'figure'),
     Output('chart-map', 'figure')],
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
        # Return empty placeholder charts on initial load
        fig_empty = px.bar(title="Click 'Generate Report' to view visualizations")
        return fig_empty, fig_empty, fig_empty, fig_empty, fig_empty

    # --- DEBUG PRINT 1 ---
    print(f"\n--- GENERATING REPORT (Click {n_clicks}) ---")
    print(f"Total rows in dataset: {len(df)}")
    print(f"Filters -> Boroughs: {selected_boroughs}, Years: {selected_years}, Search: '{search_query}'")

    # Build filter mask incrementally to avoid multiple copies
    mask = pd.Series(True, index=df.index)
    
    # 1. Filter by Borough
    if selected_boroughs:
        mask = mask & df['BOROUGH'].isin(selected_boroughs)
        print(f"Rows after Borough filter: {mask.sum():,}")

    # 2. Filter by Year
    if selected_years:
        # Use CRASH YEAR column if available (more efficient), otherwise use CRASH_DATETIME
        if 'CRASH YEAR' in df.columns:
            mask = mask & df['CRASH YEAR'].isin(selected_years)
        else:
            # Fallback to datetime parsing
            if not pd.api.types.is_datetime64_any_dtype(df['CRASH_DATETIME']):
                 df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
            mask = mask & df['CRASH_DATETIME'].dt.year.isin(selected_years)
        print(f"Rows after Year filter: {mask.sum():,}")

    # Apply borough and year filters first to reduce dataset size
    dff = df[mask].copy()
    
    # 3. Filter by Search Query (apply on already-filtered data to save memory)
    if search_query:
        # EXPANDED SEARCH: Check more columns for better results
        # We verify if the column exists before searching to prevent errors
        search_cols = [c for c in ['BOROUGH', 'CONTRIBUTING FACTOR VEHICLE 1', 'VEHICLE TYPE CODE 1', 'PERSON_TYPE', 'PERSON_INJURY'] if c in dff.columns]
        
        # Build search mask column by column on the filtered subset
        search_mask = pd.Series(False, index=dff.index)
        for col in search_cols:
            # Convert to string and search (now working on smaller filtered dataset)
            col_mask = dff[col].astype(str).str.contains(search_query, case=False, na=False)
            search_mask = search_mask | col_mask
        
        dff = dff[search_mask].copy()
        print(f"Rows after Search filter: {len(dff):,}")

    # --- DEBUG PRINT FINAL ---
    print(f"FINAL rows to plot: {len(dff)}")

    if dff.empty:
        print("!!! RESULT IS EMPTY - RETURNING BLANK CHARTS !!!")
        fig_empty = px.bar(title="No data found matching these filters.")
        return fig_empty, fig_empty, fig_empty, fig_empty, fig_empty

    # 4. Generate All Visualizations with Interactivity
    
    # 1. BAR CHART: Crash Count by Borough
    dff_counts = dff['BOROUGH'].value_counts().reset_index()
    dff_counts.columns = ['BOROUGH', 'Count']
    fig_bar = px.bar(
        dff_counts, 
        x='BOROUGH', 
        y='Count', 
        title="Crashes by Borough (Bar Chart)",
        labels={'Count': 'Number of Crashes', 'BOROUGH': 'Borough'},
        color='Count',
        color_continuous_scale='Blues'
    )
    fig_bar.update_layout(
        hovermode='x unified',
        xaxis={'categoryorder': 'total descending'},
        showlegend=False
    )
    fig_bar.update_traces(hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>')
    
    # 2. LINE CHART: Crashes by Hour of Day
    if 'CRASH HOUR' in dff.columns:
        hourly_counts = dff['CRASH HOUR'].value_counts().sort_index().reset_index()
        hourly_counts.columns = ['Hour', 'Count']
        fig_line = px.line(
            hourly_counts, 
            x='Hour', 
            y='Count', 
            title="Crashes by Hour of Day (Line Chart)",
            labels={'Count': 'Number of Crashes', 'Hour': 'Hour of Day (0-23)'},
            markers=True
        )
        fig_line.update_traces(
            line=dict(width=3), 
            marker=dict(size=8),
            hovertemplate='<b>Hour: %{x}</b><br>Crashes: %{y}<extra></extra>'
        )
        fig_line.update_xaxes(dtick=2)
        fig_line.update_layout(
            hovermode='x unified',
            xaxis_title='Hour of Day (0-23)',
            yaxis_title='Number of Crashes'
        )
    else:
        fig_line = px.line(title="No hour data available")
    
    # 3. HEATMAP: Borough x Hour of Day
    if 'CRASH HOUR' in dff.columns and 'BOROUGH' in dff.columns:
        # Create pivot table for proper heatmap
        heatmap_data = dff.groupby(['BOROUGH', 'CRASH HOUR']).size().reset_index(name='Count')
        heatmap_pivot = heatmap_data.pivot(index='BOROUGH', columns='CRASH HOUR', values='Count').fillna(0)
        
        # Use imshow for a cleaner heatmap visualization
        fig_heatmap = px.imshow(
            heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            labels=dict(x="Hour of Day", y="Borough", color="Number of Crashes"),
            title="Crash Intensity: Borough x Hour (Heatmap)",
            color_continuous_scale='YlOrRd',
            aspect='auto'
        )
        fig_heatmap.update_xaxes(dtick=2)
        fig_heatmap.update_layout(
            xaxis_title='Hour of Day (0-23)',
            yaxis_title='Borough',
            height=400
        )
        # Update hover template to show actual values
        fig_heatmap.update_traces(
            hovertemplate='<b>Borough: %{y}</b><br>Hour: %{x}<br>Crashes: %{z:.0f}<extra></extra>'
        )
    else:
        fig_heatmap = px.imshow([[0]], title="No data available for heatmap")
    
    # 4. PIE CHART: Person Injury Types Distribution
    if 'PERSON_INJURY' in dff.columns:
        injury_counts = dff['PERSON_INJURY'].value_counts().reset_index()
        injury_counts.columns = ['Injury_Type', 'Count']
        fig_pie = px.pie(
            injury_counts,
            values='Count',
            names='Injury_Type',
            title="Distribution of Injury Types (Pie Chart)",
            hole=0.4
        )
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    else:
        # Fallback: Vehicle Type distribution
        if 'VEHICLE TYPE CODE 1' in dff.columns:
            vehicle_counts = dff['VEHICLE TYPE CODE 1'].value_counts().head(8).reset_index()
            vehicle_counts.columns = ['Vehicle_Type', 'Count']
            fig_pie = px.pie(
                vehicle_counts,
                values='Count',
                names='Vehicle_Type',
                title="Top Vehicle Types Involved (Pie Chart)",
                hole=0.4
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        else:
            fig_pie = px.pie(title="No data available for pie chart")
    
    # 5. MAP: Borough Distribution (since lat/lon not available)
    borough_counts = dff['BOROUGH'].value_counts().reset_index()
    borough_counts.columns = ['BOROUGH', 'Count']
    # Create a choropleth-style visualization using borough data
    fig_map = px.bar(
        borough_counts,
        x='BOROUGH',
        y='Count',
        title="Crashes by Borough (Geographic Distribution)",
        labels={'Count': 'Number of Crashes'},
        color='Count',
        color_continuous_scale='Blues'
    )
    fig_map.update_layout(
        hovermode='x unified',
        xaxis_title='Borough',
        yaxis_title='Number of Crashes'
    )
    fig_map.update_traces(hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>')

    return fig_bar, fig_line, fig_heatmap, fig_pie, fig_map

if __name__ == '__main__':
    app.run(debug=True)