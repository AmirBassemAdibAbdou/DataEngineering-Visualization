import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import gdown
import os
import gc
import pyarrow.parquet as pq

# --- 0. MEMORY OPTIMIZATION (CRITICAL) ---
# Enable Copy-on-Write (Pandas 2.0+). 
# This prevents creating deep copies of data when filtering, 
# allowing multiple users to share the same memory footprint.
pd.options.mode.copy_on_write = True

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="NYC Motor Vehicle Collisions Report", layout="wide")

# --- 2. DATA LOADING ---
@st.cache_resource(show_spinner=False) 
def load_data():
    parquet_file = 'crashes.parquet'
    
    # A. Download if missing
    if not os.path.exists(parquet_file):
        # REPLACE THIS WITH YOUR ACTUAL FILE ID
        file_id = 'PASTE_YOUR_GOOGLE_DRIVE_FILE_ID_HERE' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, parquet_file, quiet=False)

    print("Inspecting Parquet schema...")
    
    # B. Column Pruning
    wanted_cols = [
        'BOROUGH', 'CRASH DATE', 'CRASH TIME', 'CRASH_DATETIME', 'CRASH YEAR', 'CRASH HOUR',
        'VEHICLE TYPE CODE 1', 'CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2',
        'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'ON STREET NAME',
        'PERSON_INJURY', 'PERSON_TYPE', 'PERSON_AGE', 'PERSON_SEX', 'IsDanger'
    ]
    
    try:
        parquet_schema = pq.ParquetFile(parquet_file).schema.names
        cols_to_load = [c for c in wanted_cols if c in parquet_schema]
    except Exception as e:
        print(f"Schema inspection failed: {e}")
        cols_to_load = None

    # C. Load Data
    df = pd.read_parquet(
        parquet_file, 
        columns=cols_to_load, 
        engine='pyarrow', 
        dtype_backend='pyarrow'
    )

    # D. Date Parsing & Filtering (Memory Saver)
    # Filter out very old data to save RAM (Adjust year as needed, e.g., 2019)
    MIN_YEAR = 2018 
    
    if 'CRASH YEAR' in df.columns:
        df = df[df['CRASH YEAR'] >= MIN_YEAR]
    elif 'CRASH_DATETIME' in df.columns:
        df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
        df = df[df['CRASH_DATETIME'].dt.year >= MIN_YEAR]
        df['CRASH_DATE'] = df['CRASH_DATETIME'].dt.date
        
    # E. Optimize Categories
    categorical_cols = ['BOROUGH', 'CONTRIBUTING FACTOR VEHICLE 1', 'VEHICLE TYPE CODE 1', 'PERSON_TYPE', 'PERSON_INJURY']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    gc.collect()
    return df

# Initialize Data
try:
    with st.spinner("Loading optimized dataset..."):
        df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.header("Filters")
with st.sidebar.form("filter_form"):
    
    search_query = st.text_input("Global Search", placeholder='e.g. "Brooklyn 2022"')

    # Optimized Dropdowns (Using existing dataframe without copying)
    if 'BOROUGH' in df.columns:
        unique_boroughs = sorted(df['BOROUGH'].dropna().unique().tolist())
        # Remove 'UNKNOWN' if present
        if 'UNKNOWN' in unique_boroughs: unique_boroughs.remove('UNKNOWN')
        sel_boroughs = st.multiselect("Borough:", options=unique_boroughs)
    else:
        sel_boroughs = []

    if 'CRASH YEAR' in df.columns:
        unique_years = sorted(df['CRASH YEAR'].dropna().unique())
    elif 'CRASH_DATETIME' in df.columns:
        unique_years = sorted(df['CRASH_DATETIME'].dt.year.dropna().unique())
    else:
        unique_years = []
    
    sel_years = st.multiselect("Year:", options=unique_years)

    if 'VEHICLE TYPE CODE 1' in df.columns:
        v_counts = df['VEHICLE TYPE CODE 1'].value_counts()
        popular_vehicles = sorted([str(x) for x in v_counts[v_counts >= 10000].index if pd.notna(x)])
        sel_vehicles = st.multiselect("Vehicle Type:", options=popular_vehicles)
    else:
        sel_vehicles = []

    if 'CONTRIBUTING FACTOR VEHICLE 1' in df.columns:
         if hasattr(df['CONTRIBUTING FACTOR VEHICLE 1'], 'cat'):
             unique_factors = sorted([str(x) for x in df['CONTRIBUTING FACTOR VEHICLE 1'].cat.categories if pd.notna(x)])
         else:
             unique_factors = sorted([str(x) for x in df['CONTRIBUTING FACTOR VEHICLE 1'].dropna().unique()])
         sel_factors = st.multiselect("Contributing Factor:", options=unique_factors)
    else:
        sel_factors = []

    sel_severity = st.multiselect("Collision Severity:", options=['Fatality', 'Injury', 'Property Damage Only'])

    submitted = st.form_submit_button("Generate Report")

# --- 4. FILTERING LOGIC ---
if submitted or True:
    # Due to Copy-on-Write (enabled at top), this does NOT immediately duplicate memory
    dff = df 

    if sel_boroughs:
        dff = dff[dff['BOROUGH'].isin(sel_boroughs)]

    if sel_years:
        if 'CRASH YEAR' in df.columns:
            dff = dff[dff['CRASH YEAR'].isin(sel_years)]
        elif 'CRASH_DATETIME' in df.columns:
            dff = dff[dff['CRASH_DATETIME'].dt.year.isin(sel_years)]

    if sel_vehicles:
        dff = dff[dff['VEHICLE TYPE CODE 1'].isin(sel_vehicles)]

    if sel_factors:
        dff = dff[dff['CONTRIBUTING FACTOR VEHICLE 1'].isin(sel_factors)]

    # Search Query
    if search_query:
        terms = search_query.split()
        search_cols = [c for c in ['BOROUGH', 'ON STREET NAME', 'CONTRIBUTING FACTOR VEHICLE 1', 
                                   'VEHICLE TYPE CODE 1', 'PERSON_INJURY', 'PERSON_TYPE'] if c in dff.columns]
        
        term_mask = None 
        for term in terms:
            current_term_mask = None
            
            if term.isdigit() and len(term) == 4:
                year_val = int(term)
                if 'CRASH YEAR' in dff.columns:
                    current_term_mask = (dff['CRASH YEAR'] == year_val)
                elif 'CRASH_DATETIME' in dff.columns:
                    current_term_mask = (dff['CRASH_DATETIME'].dt.year == year_val)
            
            elif term.lower() in ['injury', 'injured', 'injuries']:
                if 'NUMBER OF PERSONS INJURED' in dff.columns:
                    current_term_mask = (dff['NUMBER OF PERSONS INJURED'] > 0)
            elif term.lower() in ['fatality', 'fatal', 'killed', 'death']:
                if 'NUMBER OF PERSONS KILLED' in dff.columns:
                    current_term_mask = (dff['NUMBER OF PERSONS KILLED'] > 0)
            
            if current_term_mask is None:
                current_term_mask = pd.Series(False, index=dff.index)
                for col in search_cols:
                    try:
                        if hasattr(dff[col], 'cat'):
                            matching_cats = [cat for cat in dff[col].cat.categories if term.lower() in str(cat).lower()]
                            if matching_cats:
                                current_term_mask = current_term_mask | dff[col].isin(matching_cats)
                        else:
                            current_term_mask = current_term_mask | dff[col].astype(str).str.contains(term, case=False, na=False)
                    except:
                        pass
            
            if term_mask is None:
                term_mask = current_term_mask
            else:
                term_mask = term_mask & current_term_mask
        
        if term_mask is not None:
            dff = dff[term_mask]

    # Exclude UNKNOWN borough for plots
    if 'BOROUGH' in dff.columns:
        dff = dff[dff['BOROUGH'] != 'UNKNOWN']
        if hasattr(dff['BOROUGH'], 'cat'):
            dff['BOROUGH'] = dff['BOROUGH'].cat.remove_unused_categories()

    # --- 5. LAYOUT & VISUALIZATIONS ---
    st.title("NYC Motor Vehicle Collisions Report")
    
    # Display a warning if data was cut off
    st.caption("Note: Data restricted to 2018+ for performance optimization.")
    
    st.markdown(f"**Showing {len(dff):,} rows** based on current filters")

    if dff.empty:
        st.warning("No data found matching these filters.")
        st.stop()

    # Row 1: Bar & Line
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Crashes by Borough")
        dff_counts = dff['BOROUGH'].value_counts().reset_index()
        dff_counts.columns = ['BOROUGH', 'Count']
        fig_bar = px.bar(dff_counts, x='BOROUGH', y='Count')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Crashes by Hour")
        if 'CRASH HOUR' in dff.columns:
            hourly_counts = dff['CRASH HOUR'].value_counts().sort_index().reset_index()
            hourly_counts.columns = ['Hour', 'Count']
            fig_line = px.line(hourly_counts, x='Hour', y='Count')
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("CRASH HOUR column missing")

    # Row 2: Heatmap & Pie
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Intensity Heatmap (Hour vs Borough)")
        if 'CRASH HOUR' in dff.columns and 'BOROUGH' in dff.columns:
            heatmap_data = dff.groupby(['BOROUGH', 'CRASH HOUR']).size().reset_index(name='Count')
            heatmap_pivot = heatmap_data.pivot(index='BOROUGH', columns='CRASH HOUR', values='Count').fillna(0)
            fig_heatmap = px.imshow(heatmap_pivot, aspect='auto')
            st.plotly_chart(fig_heatmap, use_container_width=True)

    with col4:
        st.subheader("Distribution")
        if 'PERSON_INJURY' in dff.columns:
            injury_counts = dff['PERSON_INJURY'].value_counts().reset_index()
            injury_counts.columns = ['Type', 'Count']
            title = "Injury Types"
        elif 'VEHICLE TYPE CODE 1' in dff.columns:
            injury_counts = dff['VEHICLE TYPE CODE 1'].value_counts().head(8).reset_index()
            injury_counts.columns = ['Type', 'Count']
            title = "Top Vehicle Types"
        else:
            injury_counts = pd.DataFrame()
            title = "No Data"
        
        if not injury_counts.empty:
            fig_pie = px.pie(injury_counts, values='Count', names='Type', title=title)
            st.plotly_chart(fig_pie, use_container_width=True)

    # Row 3: Demographics
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Age Distribution")
        if 'PERSON_AGE' in dff.columns:
            ages = pd.to_numeric(dff['PERSON_AGE'], errors='coerce').dropna()
            if len(ages) > 0:
                bins = [0, 18, 26, 36, 46, 56, 66, np.inf]
                labels = ['0-17', '18-25', '26-35', '36-45', '46-55', '56-65', '66+']
                age_bins = pd.cut(ages, bins=bins, labels=labels, right=False)
                age_distribution = age_bins.value_counts().sort_index()
                fig_age = px.bar(x=age_distribution.index.astype(str), y=age_distribution.values, 
                                 labels={'x': 'Age Group', 'y': 'Count'})
                st.plotly_chart(fig_age, use_container_width=True)

    with col6:
        st.subheader("Sex Distribution")
        if 'PERSON_SEX' in dff.columns:
            sex_data = dff[dff['PERSON_SEX'] != 'U']
            if len(sex_data) > 0:
                sex_counts = sex_data['PERSON_SEX'].value_counts()
                fig_sex = px.bar(x=sex_counts.index.astype(str), y=sex_counts.values,
                                 labels={'x': 'Sex', 'y': 'Count'})
                st.plotly_chart(fig_sex, use_container_width=True)

    # Row 4: Danger Factors
    st.subheader("Top 5 Contributing Factors in Dangerous Collisions")
    if 'IsDanger' in dff.columns:
        dangerous = dff[dff['IsDanger'] == 1]
        if not dangerous.empty:
            dangerous_subset = dangerous[['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2']]
            all_factors = pd.melt(dangerous_subset, value_name='Factor').dropna()
            
            if not all_factors.empty:
                if hasattr(all_factors['Factor'], 'cat'):
                    all_factors = all_factors[~all_factors['Factor'].isin(['unspecified', 'Unspecified', 'UNSPECIFIED'])]
                else:
                    all_factors = all_factors[~all_factors['Factor'].astype(str).str.lower().isin(['unspecified'])]

                if not all_factors.empty:
                    top_5 = all_factors['Factor'].value_counts().nlargest(5)
                    fig_factors = px.bar(x=top_5.index.astype(str), y=top_5.values, labels={'x': 'Factor', 'y': 'Count'})
                    st.plotly_chart(fig_factors, use_container_width=True)
                else:
                    st.info("No specific factors found")
        else:
            st.info("No dangerous collisions in current selection")