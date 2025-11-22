import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import numpy as np


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
#boroughs = sorted(df['BOROUGH'].dropna().unique().tolist())
boroughs = sorted(df[df['BOROUGH'] != 'UNKNOWN']['BOROUGH'].dropna().unique().tolist())

# Extract years - use CRASH YEAR if available, otherwise parse from CRASH_DATETIME
if 'CRASH YEAR' in df.columns:
    years = sorted(df['CRASH YEAR'].dropna().astype(int).unique())
else:
    # Fallback: parse from CRASH_DATETIME
    # Use standard datetime64[ns] to avoid Windows TimeZone errors
    df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce').astype('datetime64[ns]')
    df['CRASH_DATE'] = df['CRASH_DATETIME'].dt.date
    years = sorted(df['CRASH_DATETIME'].dt.year.dropna().astype(int).unique())

# Vehicle Types (Only show items that appear at least 10,000 times)
vehicle_type_counts = df['VEHICLE TYPE CODE 1'].value_counts()
vehicle_types = sorted([str(x) for x in vehicle_type_counts[vehicle_type_counts >= 10000].index if pd.notna(x)])

# Contributing Factors
if df['CONTRIBUTING FACTOR VEHICLE 1'].dtype.name == 'category':
    factors = sorted([str(x) for x in df['CONTRIBUTING FACTOR VEHICLE 1'].cat.categories if pd.notna(x)])
else:
    factors = sorted([str(x) for x in df['CONTRIBUTING FACTOR VEHICLE 1'].dropna().unique()])

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
                
                # Row 3: Age Distribution and Sex Distribution
                html.Div([
                    dcc.Graph(id='chart-age', style={'display': 'inline-block', 'width': '48%'}),
                    dcc.Graph(id='chart-sex', style={'display': 'inline-block', 'width': '48%'})
                ], style={'width': '100%', 'marginTop': '20px'}),
                
                # Row 4: Top Contributing Factors (full width)
                html.Div([
                    dcc.Graph(id='chart-factors', style={'width': '100%'})
                ], style={'width': '100%', 'marginTop': '20px'}),
            ])
            
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'})
])

# 4. Define the Callback - Returns all chart types
@app.callback(
    [Output('chart-bar', 'figure'),
     Output('chart-line', 'figure'),
     Output('chart-heatmap', 'figure'),
     Output('chart-pie', 'figure'),
     Output('chart-age', 'figure'),
     Output('chart-sex', 'figure'),
     Output('chart-factors', 'figure')],
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
        return fig_empty, fig_empty, fig_empty, fig_empty, fig_empty, fig_empty, fig_empty

    print(f"\n--- GENERATING REPORT (Click {n_clicks}) ---")
    
    # Build filter mask incrementally
    mask = pd.Series(True, index=df.index)
    
    # 1. Filter by Borough
    if sel_boroughs:
        mask = mask & df['BOROUGH'].isin(sel_boroughs)

    # 2. Filter by Year
    if sel_years:
        if 'CRASH YEAR' in df.columns:
            mask = mask & df['CRASH YEAR'].isin(sel_years)
        else:
            if not pd.api.types.is_datetime64_any_dtype(df['CRASH_DATETIME']):
                 df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
            mask = mask & df['CRASH_DATETIME'].dt.year.isin(sel_years)

    # Apply filters first
    dff = df[mask].copy()
    
    # Exclude 'UNKNOWN' borough from all visualizations
    if 'BOROUGH' in dff.columns:
        dff = dff[dff['BOROUGH'] != 'UNKNOWN']
        dff['BOROUGH'] = dff['BOROUGH'].cat.remove_unused_categories()

    # 3. Filter by Search Query (FIXED LOGIC)
    if search_query:
        # Split query into terms (e.g. "Brooklyn 2022" -> ["Brooklyn", "2022"])
        terms = search_query.split()
        
        # Define columns to search text in
        # ADDED: Contributing Factors, Vehicle Types, and Injury related columns
        search_cols = [
            c for c in [
                'BOROUGH', 'ON STREET NAME', 
                'CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2',
                'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2',
                'PERSON_INJURY', 'PERSON_TYPE'
            ] if c in dff.columns
        ]
        
        for term in terms:
            # A. Smart Year Filter (if term is "2022")
            if term.isdigit() and len(term) == 4:
                year_val = int(term)
                if 'CRASH YEAR' in dff.columns:
                    dff = dff[dff['CRASH YEAR'] == year_val]
                else:
                    dff = dff[dff['CRASH_DATETIME'].dt.year == year_val]
                continue 

            # B. Smart Severity Filter (Handle "Injury" or "Fatality" queries)
            # This allows searching for "Fatality" even if the text column doesn't exist
            if term.lower() in ['injury', 'injured', 'injuries']:
                dff = dff[dff['NUMBER OF PERSONS INJURED'] > 0]
                continue
            if term.lower() in ['fatality', 'fatal', 'killed', 'death']:
                dff = dff[dff['NUMBER OF PERSONS KILLED'] > 0]
                continue

            # C. Text Search in Columns
            # We create a mask for this term across all valid columns
            term_mask = pd.Series(False, index=dff.index)
            for col in search_cols:
                try:
                    # Handle categorical columns more efficiently
                    if dff[col].dtype.name == 'category':
                        # For categorical, check if term matches any category name
                        matching_cats = [cat for cat in dff[col].cat.categories if term.lower() in str(cat).lower()]
                        if matching_cats:
                            term_mask = term_mask | dff[col].isin(matching_cats)
                    else:
                        # For regular columns, convert to string and search
                        term_mask = term_mask | dff[col].astype(str).str.contains(term, case=False, na=False)
                except Exception:
                    continue
            
            # Filter data to keep only rows that have this term
            dff = dff[term_mask]

    print(f"FINAL rows to plot: {len(dff)}")

    if dff.empty:
        fig_empty = px.bar(title="No data found matching these filters.")
        return fig_empty, fig_empty, fig_empty, fig_empty, fig_empty, fig_empty, fig_empty

    # 4. Generate Visualizations
    
    # Bar Chart
    dff_counts = dff['BOROUGH'].value_counts().reset_index()
    dff_counts.columns = ['BOROUGH', 'Count']
    fig_bar = px.bar(dff_counts, x='BOROUGH', y='Count', title="Crashes by Borough")
    
    # Line Chart
    if 'CRASH HOUR' in dff.columns:
        hourly_counts = dff['CRASH HOUR'].value_counts().sort_index().reset_index()
        hourly_counts.columns = ['Hour', 'Count']
        fig_line = px.line(hourly_counts, x='Hour', y='Count', title="Crashes by Hour")
    else:
        fig_line = px.line(title="No hour data available")
    
    # Heatmap
    if 'CRASH HOUR' in dff.columns and 'BOROUGH' in dff.columns:
        heatmap_data = dff.groupby(['BOROUGH', 'CRASH HOUR']).size().reset_index(name='Count')
        heatmap_pivot = heatmap_data.pivot(index='BOROUGH', columns='CRASH HOUR', values='Count').fillna(0)
        fig_heatmap = px.imshow(heatmap_pivot, title="Intensity Heatmap")
    else:
        fig_heatmap = px.imshow([[0]], title="No data available")
    
    # Pie Chart
    if 'PERSON_INJURY' in dff.columns:
        injury_counts = dff['PERSON_INJURY'].value_counts().reset_index()
        injury_counts.columns = ['Injury_Type', 'Count']
        fig_pie = px.pie(injury_counts, values='Count', names='Injury_Type', title="Injury Types")
    else:
        if 'VEHICLE TYPE CODE 1' in dff.columns:
            vehicle_counts = dff['VEHICLE TYPE CODE 1'].value_counts().head(8).reset_index()
            vehicle_counts.columns = ['Vehicle_Type', 'Count']
            fig_pie = px.pie(vehicle_counts, values='Count', names='Vehicle_Type', title="Vehicle Types")
        else:
            fig_pie = px.pie(title="No data available")
    
    # Age Distribution Chart
    if 'PERSON_AGE' in dff.columns:
        ages = pd.to_numeric(dff['PERSON_AGE'], errors='coerce').dropna()
        if len(ages) > 0:
            bins = [0, 18, 26, 36, 46, 56, 66, np.inf]
            labels = ['0-17', '18-25', '26-35', '36-45', '46-55', '56-65', '66+']
            age_bins = pd.cut(ages, bins=bins, labels=labels, right=False)
            age_distribution = age_bins.value_counts().sort_index()
            fig_age = px.bar(
                x=age_distribution.index.astype(str), 
                y=age_distribution.values,
                title="Distribution of Persons Involved in Collisions by Age Group",
                labels={'x': 'Age Group', 'y': 'Number of Persons (Count)'}
            )
            fig_age.update_layout(xaxis_title='Age Group', yaxis_title='Number of Persons (Count)')
        else:
            fig_age = px.bar(title="No age data available")
    else:
        fig_age = px.bar(title="No age data available")
    
    # Sex Distribution Chart
    if 'PERSON_SEX' in dff.columns:
        sex_data = dff[dff['PERSON_SEX'] != 'U']
        if len(sex_data) > 0:
            sex_counts = sex_data['PERSON_SEX'].value_counts()
            fig_sex = px.bar(
                x=sex_counts.index.astype(str),
                y=sex_counts.values,
                title="Number of Persons by Sex",
                labels={'x': 'Sex', 'y': 'Count'}
            )
            fig_sex.update_layout(xaxis_title='Sex', yaxis_title='Count')
        else:
            fig_sex = px.bar(title="No sex data available")
    else:
        fig_sex = px.bar(title="No sex data available")
    
    # Top 5 Contributing Factors in Dangerous Collisions
    if 'IsDanger' in dff.columns:
        dangerous_collisions = dff[dff['IsDanger'] == 1]
        if len(dangerous_collisions) > 0:
            all_factors = pd.melt(
                dangerous_collisions, 
                value_vars=['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2'], 
                value_name='Factor'
            )
            # Filter out 'unspecified' more efficiently
            all_factors = all_factors[all_factors['Factor'].notna()]
            # Use categorical comparison if possible, otherwise convert only Factor column
            if all_factors['Factor'].dtype.name == 'category':
                all_factors = all_factors[~all_factors['Factor'].isin(['unspecified', 'Unspecified', 'UNSPECIFIED'])]
            else:
                factor_str = all_factors['Factor'].astype(str).str.lower()
                all_factors = all_factors[factor_str != 'unspecified']
            
            if len(all_factors) > 0:
                top_5_factors = all_factors['Factor'].value_counts().nlargest(5)
                fig_factors = px.bar(
                    x=top_5_factors.index.astype(str),
                    y=top_5_factors.values,
                    title="Top 5 Contributing Factors in Dangerous Collisions",
                    labels={'x': 'Contributing Factor', 'y': 'Count'}
                )
                fig_factors.update_layout(
                    xaxis_title='Contributing Factor',
                    yaxis_title='Count',
                    xaxis={'tickangle': -45}
                )
            else:
                fig_factors = px.bar(title="No contributing factor data available")
        else:
            fig_factors = px.bar(title="No dangerous collisions found")
    else:
        fig_factors = px.bar(title="No danger indicator available")
    
    return fig_bar, fig_line, fig_heatmap, fig_pie, fig_age, fig_sex, fig_factors

if __name__ == '__main__':
    app.run(debug=True)