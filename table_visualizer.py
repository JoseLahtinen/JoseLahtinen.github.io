import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from typing import Dict, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Page configuration
st.set_page_config(
    page_title="Table Data Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Statistics Finland API Configuration
SF_API_BASE = "https://pxdata.stat.fi/PXWeb/api/v1/en/StatFin"

# Cache API calls
@st.cache_data(ttl=3600)
def fetch_stat_fin_databases() -> Dict:
    """Fetch available databases from Statistics Finland API"""
    try:
        with st.spinner("Loading Statistics Finland databases..."):
            # use a session with retry to handle transient 429/5xx responses
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            response = session.get(SF_API_BASE, timeout=10)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error fetching databases: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_stat_fin_tables(db_id: str) -> Dict:
    """Fetch available tables from a Statistics Finland database"""
    try:
        url = f"{SF_API_BASE}/{db_id}"
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []


@st.cache_data(ttl=3600)
def fetch_stat_fin_table_metadata(db_id: str, table_id: str) -> Dict:
    """Fetch table metadata (variables and categories) from Statistics Finland"""
    try:
        url = f"{SF_API_BASE}/{db_id}/{table_id}"
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching table metadata: {e}")
        return {}


@st.cache_data(ttl=3600)
def fetch_stat_fin_table_data_with_query(db_id: str, table_id: str, query: Dict) -> Optional[pd.DataFrame]:
    """Fetch table data using a prepared PXWeb query dict"""
    try:
        with st.spinner("Loading data..."):
            data_url = f"{SF_API_BASE}/{db_id}/{table_id}"
            headers = {"Content-Type": "application/json"}
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            data_response = session.post(
                data_url,
                json=query,
                headers=headers,
                timeout=30
            )
            data_response.raise_for_status()
            data = data_response.json()

            if "data" in data and "dimension" in data:
                return parse_pxweb_data(data)
            else:
                st.warning("No data returned for that query")
                return None
    except Exception as e:
        st.error(f"Error fetching table data: {e}")
        return None


def estimate_query_size(variable_selections: Dict[str, List[str]]) -> int:
    """Estimate number of cells returned by a query (product of selection lengths)."""
    size = 1
    for v in variable_selections.values():
        if not v:
            continue
        size *= len(v)
    return size


def chunked_fetch(db_id: str, table_id: str, variable_selections: Dict[str, List[str]], max_chunk_cells: int = 2000) -> Optional[pd.DataFrame]:
    """Fetch data by splitting the largest variable into batches so each request stays under max_chunk_cells."""
    # Determine sizes
    sizes = {k: len(v) if v else 1 for k, v in variable_selections.items()}
    if not sizes:
        return fetch_stat_fin_table_data_with_query(db_id, table_id, {"response": {"format": "json"}})

    # Choose split variable as the one with largest cardinality
    split_var = max(sizes.items(), key=lambda x: x[1])[0]
    total = estimate_query_size(variable_selections)

    # compute batch size for split_var
    other_product = max(1, total // sizes[split_var])
    if other_product == 0:
        other_product = 1
    values = variable_selections[split_var]

    # Determine chunk length
    chunk_len = max(1, max_chunk_cells // other_product)

    dfs = []
    progress = st.progress(0)
    try:
        for i in range(0, len(values), chunk_len):
            batch_vals = values[i:i+chunk_len]
            q = {"response": {"format": "json"}}
            for code, sel in variable_selections.items():
                if code == split_var:
                    q[code] = batch_vals
                else:
                    q[code] = sel
            df_part = fetch_stat_fin_table_data_with_query(db_id, table_id, q)
            if df_part is not None and not df_part.empty:
                dfs.append(df_part)
            progress.progress(min(1.0, (i+chunk_len) / len(values)))
        progress.empty()
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return None
    except Exception as e:
        st.error(f"Chunked fetch failed: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_stat_fin_table_data(db_id: str, table_id: str) -> Optional[pd.DataFrame]:
    """Fetch actual data from Statistics Finland"""
    try:
        with st.spinner(f"Loading {table_id}..."):
            # First, get the table metadata
            metadata_url = f"{SF_API_BASE}/{db_id}/{table_id}"
            metadata_response = requests.get(metadata_url, timeout=10)
            metadata_response.raise_for_status()
            metadata = metadata_response.json()
            
            # Prepare the query - get first variable selection for each dimension
            query = {
                "response": {
                    "format": "json"
                }
            }
            
            # Add variable selections (default: first value for each variable)
            if "variables" in metadata:
                for var in metadata["variables"]:
                    if var.get("values"):
                        query[var["code"]] = [var["values"][0]]
            
            # Request data with the query
            data_url = f"{SF_API_BASE}/{db_id}/{table_id}"
            headers = {"Content-Type": "application/json"}
            data_response = requests.post(
                data_url, 
                json=query, 
                headers=headers,
                timeout=15
            )
            data_response.raise_for_status()
            data = data_response.json()
            
            # Convert PXWeb response to DataFrame
            if "data" in data and "dimension" in data:
                return parse_pxweb_data(data)
            else:
                st.warning("No data format available")
                return None
                
    except Exception as e:
        st.error(f"Error fetching table data: {e}")
        return None

def parse_pxweb_data(pxweb_response: Dict) -> pd.DataFrame:
    """Parse PXWeb API JSON response into a pandas DataFrame"""
    try:
        data_values = pxweb_response.get("data", [])
        dimensions = pxweb_response.get("dimension", {})
        
        # Build the column structure from dimensions
        rows = []
        for item in data_values:
            row = {}
            
            # Add dimension values
            key_parts = item.get("key", [])
            for i, (dim_name, dim_info) in enumerate(dimensions.items()):
                if i < len(key_parts):
                    key = key_parts[i]
                    # Prefer label mapping if available, otherwise fall back to key
                    cat = dim_info.get("category", {})
                    labels = cat.get("label") or cat.get("labels") or {}
                    # Labels keys may be strings or ints — try both
                    label_val = labels.get(str(key)) if isinstance(labels, dict) else None
                    if label_val is None:
                        label_val = labels.get(key) if isinstance(labels, dict) else None
                    row[dim_name] = label_val if label_val is not None else key
            
            # Add the value
            value = item.get("value")
            row["Value"] = value if value is not None else 0
            rows.append(row)
        
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return pd.DataFrame()

# Title
st.title("📊 Table Data Visualizer")
st.markdown("Interactive table data visualization and analysis tool")

# Sidebar for data source selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Select data source:",
    ["Sample Data", "Upload CSV", "Manual Input", "Statistics Finland"]
)

# Initialize dataframe
df = None

if data_source == "Sample Data":
    # Sample dataset
    df = pd.DataFrame({
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Webcam'],
        'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Electronics'],
        'Sales': [15000, 2500, 3200, 8500, 4200, 3800],
        'Units Sold': [45, 320, 280, 65, 150, 120],
        'Rating': [4.8, 4.5, 4.3, 4.7, 4.6, 4.4],
        'In Stock': [True, True, False, True, True, False]
    })

elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file to visualize your data")

elif data_source == "Manual Input":
    st.subheader("Enter Table Data")
    col1, col2 = st.columns(2)
    
    with col1:
        rows = st.number_input("Number of rows:", min_value=1, max_value=100, value=5)
        cols = st.number_input("Number of columns:", min_value=1, max_value=20, value=3)
    
    # Create input fields for manual data entry
    data = {}
    for i in range(cols):
        col_name = st.text_input(f"Column {i+1} name:", value=f"Column_{i+1}", key=f"col_name_{i}")
        data[col_name] = []
        for j in range(rows):
            value = st.text_input(f"{col_name} - Row {j+1}:", key=f"cell_{i}_{j}")
            data[col_name].append(value)
    
    if any(data.values()):
        df = pd.DataFrame(data)

elif data_source == "Statistics Finland":
    st.subheader("Statistics Finland Data Browser")
    
    # Fetch available databases
    databases = fetch_stat_fin_databases()
    
    if databases:
        # Create a mapping of database names to IDs
        db_dict = {item["text"]: item["id"] for item in databases if item.get("type") == "l"}
        
        selected_db_name = st.selectbox(
            "Select Database:",
            options=sorted(db_dict.keys()),
            key="sf_db_select"
        )
        
        if selected_db_name:
            selected_db_id = db_dict[selected_db_name]
            
            # Fetch tables from selected database
            tables = fetch_stat_fin_tables(selected_db_id)
            
            if tables:
                # Create a mapping of table names to IDs
                table_dict = {}
                for item in tables:
                    if item.get("type") == "l":
                        # Use text as display name, id for requests
                        table_dict[item["text"]] = item["id"]
                
                if table_dict:
                    selected_table_name = st.selectbox(
                        "Select Table:",
                        options=sorted(table_dict.keys()),
                        key="sf_table_select"
                    )
                    
                    if selected_table_name:
                        selected_table_id = table_dict[selected_table_name]
                        
                        # Fetch metadata for the selected table to build selectors
                        metadata = fetch_stat_fin_table_metadata(selected_db_id, selected_table_id)
                        variable_selections = {}

                        select_all_opt = st.checkbox("Select all categories for each variable (may return large datasets)", value=False)

                        if metadata and metadata.get("variables"):
                            st.markdown("**Choose variable selections**")
                            for var in metadata["variables"]:
                                code = var.get("code")
                                name = var.get("text") or code
                                # Try common keys for category values
                                options = []
                                if var.get("values"):
                                    options = var.get("values")
                                elif var.get("valueTexts") and isinstance(var.get("valueTexts"), dict):
                                    options = list(var.get("valueTexts").keys())
                                elif var.get("valueTexts") and isinstance(var.get("valueTexts"), list):
                                    options = var.get("valueTexts")
                                elif var.get("values") is None and var.get("valueTexts") is None:
                                    # Fallback: try categories from metadata dimension
                                    dim = metadata.get("dimension", {}).get(code, {})
                                    cat = dim.get("category", {})
                                    labels = cat.get("label") or cat.get("labels") or {}
                                    if isinstance(labels, dict):
                                        options = list(labels.keys())

                                # Ensure options is a list of strings
                                options = [str(o) for o in (options or [])]
                                if not options:
                                    # If no options, skip this variable
                                    continue

                                default = options if select_all_opt else [options[0]]
                                sel = st.multiselect(f"{name} ({code})", options=options, default=default, key=f"var_{code}")
                                variable_selections[code] = sel

                        # Now provide preview and size/ chunking controls
                        estimated_cells = estimate_query_size(variable_selections)
                        st.write(f"Estimated result size: **{estimated_cells}** cells")
                        large_threshold = 5000
                        if estimated_cells > large_threshold:
                            st.warning("This query is large and may take a long time or be rate-limited. Consider loading a preview or using chunked loading.")

                        # Chunked loading options
                        chunk_opts = st.expander("Chunked loading options (advanced)")
                        with chunk_opts:
                            max_chunk_cells = st.number_input("Max cells per request (chunk size):", min_value=500, max_value=20000, value=2000, step=500)
                            enable_chunked = st.checkbox("Enable chunked loading for large queries", value=True)

                        # Preview button: fetch a tiny preview using first value of each variable
                        if st.button("Load Preview (small)", key="load_preview"):
                            preview_query = {"response": {"format": "json"}}
                            for code, vals in variable_selections.items():
                                if vals:
                                    preview_query[code] = [vals[0]]
                            preview_df = fetch_stat_fin_table_data_with_query(selected_db_id, selected_table_id, preview_query)
                            if preview_df is not None:
                                st.dataframe(preview_df.head(100))

                        # Build and run the full query when requested
                        if st.button("Load Data", key="load_sf_data"):
                            query: Dict = {"response": {"format": "json"}}
                            for code, selected_vals in variable_selections.items():
                                if selected_vals:
                                    query[code] = selected_vals

                            if not variable_selections:
                                st.info("No variable selections available for this table. Attempting to load default data.")
                                df = fetch_stat_fin_table_data_with_query(selected_db_id, selected_table_id, {"response": {"format": "json"}})
                            else:
                                # If estimated size is big and chunked loading enabled, use chunked_fetch
                                if estimated_cells > large_threshold and enable_chunked:
                                    st.info("Loading data in chunks...")
                                    df = chunked_fetch(selected_db_id, selected_table_id, variable_selections, max_chunk_cells=int(max_chunk_cells))
                                else:
                                    df = fetch_stat_fin_table_data_with_query(selected_db_id, selected_table_id, query)
                else:
                    st.info("No tables available in this database")
            else:
                st.info("Loading tables...")
    else:
        st.warning("Unable to connect to Statistics Finland API. Please check your internet connection.")

# Main content
if df is not None and not df.empty:
    st.success(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Data Display Options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_stats = st.checkbox("Show Statistics", value=True)
    with col2:
        show_chart = st.checkbox("Show Chart", value=True)
    with col3:
        show_filters = st.checkbox("Show Filters", value=True)
    
    # Filters
    if show_filters:
        st.subheader("Filters")
        filtered_df = df.copy()
        
        filter_cols = st.multiselect(
            "Filter by columns:",
            options=df.columns.tolist(),
            default=[]
        )
        
        for col in filter_cols:
            if df[col].dtype == 'object':
                selected_values = st.multiselect(
                    f"Select {col}:",
                    options=df[col].unique().tolist(),
                    default=df[col].unique().tolist()
                )
                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
            else:
                min_val, max_val = st.slider(
                    f"Range for {col}:",
                    min_value=float(df[col].min()),
                    max_value=float(df[col].max()),
                    value=(float(df[col].min()), float(df[col].max()))
                )
                filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
    else:
        filtered_df = df.copy()
    
    # Display Table
    st.subheader("Data Table")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Statistics
    if show_stats:
        st.subheader("Statistics")
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            for idx, col in enumerate(numeric_cols[:3]):
                if idx % 3 == 0:
                    with stats_col1:
                        st.metric(f"{col} - Mean", f"{filtered_df[col].mean():.2f}")
                elif idx % 3 == 1:
                    with stats_col2:
                        st.metric(f"{col} - Mean", f"{filtered_df[col].mean():.2f}")
                else:
                    with stats_col3:
                        st.metric(f"{col} - Mean", f"{filtered_df[col].mean():.2f}")
            
            # Detailed statistics table
            st.write(filtered_df.describe())
    
    # Charts
    if show_chart and len(numeric_cols) > 0:
        st.subheader("Visualization")
        
        chart_col1, chart_col2 = st.columns(2)
        
        # Get numeric columns for charting
        numeric_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = filtered_df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_columns:
            with chart_col1:
                x_col = st.selectbox("X-axis:", options=filtered_df.columns, index=0, key="x_axis")
                y_col = st.selectbox("Y-axis:", options=numeric_columns, key="y_axis")
                
                if x_col and y_col:
                    fig = px.scatter(filtered_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                if numeric_columns:
                    chart_col = st.selectbox("Column to visualize:", options=numeric_columns, key="chart_col")
                    fig = px.histogram(filtered_df, x=chart_col, title=f"Distribution of {chart_col}")
                    st.plotly_chart(fig, use_container_width=True)

else:
    if data_source != "Sample Data":
        st.warning("No data available. Please upload a file or select Sample Data.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/)")
