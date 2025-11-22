import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import gdown
import os
import gc # Garbage collection to free memory

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="NYC Motor Vehicle Collisions Report", layout="wide")

# --- 2. DATA LOADING (Optimized for Memory) ---
# CHANGED: Use cache_resource for large static datasets (Shared memory, no copying)
@st.cache_resource(show_spinner=False) 
def load_data():
    parquet_file = 'crashes.parquet'
    
    # A. Download if missing
    if not os.path.exists(parquet_file):
        # REPLACE THIS WITH YOUR ACTUAL FILE ID
        file_id = '1oVO3wrLpqReuu1aS5gukOjCniDNFBeoq' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, parquet_file, quiet=False)

    # B. Load Data with Memory Optimizations
    print("Loading parquet file into memory...")
    # engine='pyarrow' and dtype_backend='pyarrow' help reduce RAM usage significantly
    df = pd.read_parquet(parquet_file, engine='pyarrow', dtype_backend='pyarrow')

    # C. Optimize dtypes
    # We define categories explicitly to ensure standard pandas handling where needed
    categorical_cols = ['BOROUGH', 'CONTRIBUTING FACTOR VEHICLE 1', 'VEHICLE TYPE CODE 1', 'PERSON_TYPE', 'PERSON_INJURY']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # D. Date Parsing
    if 'CRASH YEAR' not in df.columns:
        # Use 'coerce' to handle errors and keep memory tight
        df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
        df['CRASH_DATE'] = df['CRASH_DATETIME'].dt.date

    # Force garbage collection to clear download buffers immediately
    gc.collect()
    
    return df

# Initialize Data
try:
    with st.spinner("Loading 277MB Dataset... (This happens once)"):
        df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.header("Filters")
with st.sidebar.form("filter_form"):
    
    # Search Box
    search_query = st.text_input("Global Search", placeholder='e.g. "Brooklyn 2022"')

    # Borough Filter
    if 'BOROUGH' in df.columns:
        # Filter out UNKNOWN/NaN for the dropdown list
        unique_boroughs = sorted(df[df['BOROUGH'].notna() & (df['BOROUGH'] != 'UNKNOWN')]['BOROUGH'].unique().tolist())
        sel_boroughs = st.multiselect("Borough:", options=unique_boroughs)
    else:
        sel_boroughs = []

    # Year Filter
    if 'CRASH YEAR' in df.columns:
        unique_years = sorted(df['CRASH YEAR'].dropna().unique())
    else:
        unique_years = sorted(df['CRASH_DATETIME'].dt.year.dropna().unique())
    
    sel_years = st.multiselect("Year:", options=unique_years)

    # Vehicle Type Filter (Top items only > 10000)
    if 'VEHICLE TYPE CODE 1' in df.columns:
        v_counts = df['VEHICLE TYPE CODE 1'].value_counts()
        popular_vehicles = sorted([str(x) for x in v_counts[v_counts >= 10000].index if pd.notna(x)])
        sel_vehicles = st.multiselect("Vehicle Type:", options=popular_vehicles)
    else:
        sel_vehicles = []

    # Contributing Factor Filter
    if 'CONTRIBUTING FACTOR VEHICLE 1' in df.columns:
         if df['CONTRIBUTING FACTOR VEHICLE 1'].dtype.name == 'category':
             unique_factors = sorted([str(x) for x in df['CONTRIBUTING FACTOR VEHICLE 1'].cat.categories if pd.notna(x)])
         else:
             unique_factors = sorted([str(x) for x in df['CONTRIBUTING FACTOR VEHICLE 1'].dropna().unique()])
         sel_factors = st.multiselect("Contributing Factor:", options=unique_factors)
    else:
        sel_factors = []

    # Severity Filter
    sel_severity = st.multiselect("Collision Severity:", options=['Fatality', 'Injury', 'Property Damage Only'])

    submitted = st.form_submit_button("Generate Report")

# --- 4. FILTERING LOGIC ---
# Only run filtering if the user submitted or if it's the first load
if submitted or True:
    # mask = pd.Series(True, index=df.index) # High memory usage to create a full boolean series
    # Better strategy: Chained filtering (filters data progressively)
    
    dff = df # Start with full reference (no copy yet)

    # A. Borough
    if sel_boroughs:
        dff = dff[dff['BOROUGH'].isin(sel_boroughs)]

    # B. Year
    if sel_years:
        if 'CRASH YEAR' in df.columns:
            dff = dff[dff['CRASH YEAR'].isin(sel_years)]
        else:
            dff = dff[dff['CRASH_DATETIME'].dt.year.isin(sel_years)]

    # C. Vehicle
    if sel_vehicles:
        dff = dff[dff['VEHICLE TYPE CODE 1'].isin(sel_vehicles)]

    # D. Factors
    if sel_factors:
        dff = dff[dff['CONTRIBUTING FACTOR VEHICLE 1'].isin(sel_factors)]

    # E. Search Query
    if search_query:
        terms = search_query.split()
        search_cols = [c for c in ['BOROUGH', 'ON STREET NAME', 'CONTRIBUTING FACTOR VEHICLE 1', 
                                   'VEHICLE TYPE CODE 1', 'PERSON_INJURY', 'PERSON_TYPE'] if c in dff.columns]
        
        term_mask = None 
        for term in terms:
            # Year Smart Search
            if term.isdigit() and len(term) == 4:
                year_val = int(term)
                if 'CRASH YEAR' in dff.columns:
                    current_term_mask = (dff['CRASH YEAR'] == year_val)
                else:
                    current_term_mask = (dff['CRASH_DATETIME'].dt.year == year_val)
            
            # Severity Smart Search
            elif term.lower() in ['injury', 'injured', 'injuries']:
                current_term_mask = (dff['NUMBER OF PERSONS INJURED'] > 0)
            elif term.lower() in ['fatality', 'fatal', 'killed', 'death']:
                current_term_mask = (dff['NUMBER OF PERSONS KILLED'] > 0)
            
            # Text Search
            else:
                current_term_mask = pd.Series(False, index=dff.index)
                for col in search_cols:
                    try:
                        if dff[col].dtype.name == 'category':
                            # Check categories first (faster)
                            matching_cats = [cat for cat in dff[col].cat.categories if term.lower() in str(cat).lower()]
                            if matching_cats:
                                current_term_mask = current_term_mask | dff[col].isin(matching_cats)
                        else:
                            current_term_mask = current_term_mask | dff[col].astype(str).str.contains(term, case=False, na=False)
                    except:
                        pass
            
            # Combine masks
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
            # Optimization: Select only necessary columns before melting
            dangerous_subset = dangerous[['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2']]
            all_factors = pd.melt(dangerous_subset, value_name='Factor').dropna()
            
            if not all_factors.empty:
                # Check type safely
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