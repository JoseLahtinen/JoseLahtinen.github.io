import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from typing import Dict, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import deque
from time import time
import time as time_module
import io
from PIL import Image
import math

# Statistics Finland API Limits
SF_API_LIMITS = {
    "maxValues": 120000,
    "maxCells": 120000,
    "maxCalls": 40,
    "timeWindow": 60,  # seconds
    "CORS": True
}

class RateLimiter:
    """Simple rate limiter that tracks call timestamps."""
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times = deque()

    def is_allowed(self) -> bool:
        now = time()
        while self.call_times and self.call_times[0] < now - self.time_window:
            self.call_times.popleft()
        return len(self.call_times) < self.max_calls

    def record_call(self):
        self.call_times.append(time())

    def wait_if_needed(self):
        while not self.is_allowed():
            oldest = self.call_times[0] if self.call_times else time()
            sleep_time = (oldest + self.time_window) - time()
            if sleep_time > 0:
                time_module.sleep(min(sleep_time + 0.1, 1.0))

    def get_remaining_calls(self) -> int:
        now = time()
        while self.call_times and self.call_times[0] < now - self.time_window:
            self.call_times.popleft()
        return max(0, self.max_calls - len(self.call_times))

# Global rate limiter instance
sf_rate_limiter = RateLimiter(SF_API_LIMITS["maxCalls"], SF_API_LIMITS["timeWindow"])

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
            sf_rate_limiter.wait_if_needed()
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            response = session.get(SF_API_BASE, timeout=10)
            response.raise_for_status()
            sf_rate_limiter.record_call()
            return response.json()
    except Exception as e:
        st.error(f"Error fetching databases: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_stat_fin_tables(db_id: str) -> Dict:
    """Fetch available tables from a Statistics Finland database"""
    try:
        url = f"{SF_API_BASE}/{db_id}"
        sf_rate_limiter.wait_if_needed()
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=10)
        response.raise_for_status()
        sf_rate_limiter.record_call()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []


@st.cache_data(ttl=3600)
def fetch_stat_fin_table_metadata(db_id: str, table_id: str) -> Dict:
    """Fetch table metadata (variables and categories) from Statistics Finland"""
    try:
        url = f"{SF_API_BASE}/{db_id}/{table_id}"
        sf_rate_limiter.wait_if_needed()
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=10)
        response.raise_for_status()
        sf_rate_limiter.record_call()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching table metadata: {e}")
        return {}


@st.cache_data(ttl=3600)
def fetch_stat_fin_table_data_with_query(db_id: str, table_id: str, variable_selections: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
    """Fetch table data using variable selections. Builds correct PXWeb JSON query format."""
    try:
        with st.spinner("Loading data..."):
            sf_rate_limiter.wait_if_needed()
            data_url = f"{SF_API_BASE}/{db_id}/{table_id}"
            
            # Build correct PXWeb query format
            query_list = []
            for code, values in variable_selections.items():
                if values:  # Only include non-empty selections
                    query_list.append({
                        "code": code,
                        "selection": {
                            "filter": "item",
                            "values": values
                        }
                    })
            
            query = {
                "query": query_list,
                "response": {"format": "json"}
            }
            
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
            sf_rate_limiter.record_call()
            data = data_response.json()

            if "data" in data:
                return parse_pxweb_json_data(data)
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


def validate_query_size(variable_selections: Dict[str, List[str]]) -> tuple[bool, str]:
    """Validate query against API limits. Returns (is_valid, message)."""
    size = estimate_query_size(variable_selections)
    max_cells = SF_API_LIMITS["maxCells"]
    
    if size > max_cells:
        return False, f"Query size ({size} cells) exceeds API limit ({max_cells} cells). Please reduce selections."
    return True, f"Query size: {size} cells (limit: {max_cells})"


def chunked_fetch(db_id: str, table_id: str, variable_selections: Dict[str, List[str]], max_chunk_cells: int = 2000) -> Optional[pd.DataFrame]:
    """Fetch data by splitting the largest variable into batches so each request stays under max_chunk_cells and API limits."""
    # Determine sizes
    sizes = {k: len(v) if v else 1 for k, v in variable_selections.items()}
    if not sizes:
        return fetch_stat_fin_table_data_with_query(db_id, table_id, {})

    # Choose split variable as the one with largest cardinality
    split_var = max(sizes.items(), key=lambda x: x[1])[0]
    total = estimate_query_size(variable_selections)

    # Ensure chunk size respects API limits (120k cells)
    api_max_cells = SF_API_LIMITS["maxCells"]
    max_chunk_cells = min(max_chunk_cells, api_max_cells)

    # compute batch size for split_var
    other_product = max(1, total // sizes[split_var])
    if other_product == 0:
        other_product = 1
    values = variable_selections[split_var]

    # Determine chunk length
    chunk_len = max(1, max_chunk_cells // other_product)

    dfs = []
    progress = st.progress(0)
    status_text = st.empty()
    try:
        num_chunks = (len(values) + chunk_len - 1) // chunk_len
        for i, chunk_idx in enumerate(range(0, len(values), chunk_len)):
            batch_vals = values[chunk_idx:chunk_idx+chunk_len]
            
            # Build variable selections for this chunk
            chunk_selections = dict(variable_selections)
            chunk_selections[split_var] = batch_vals
            
            # Show status
            status_text.text(f"Loading chunk {i+1}/{num_chunks}...")
            
            df_part = fetch_stat_fin_table_data_with_query(db_id, table_id, chunk_selections)
            if df_part is not None and not df_part.empty:
                dfs.append(df_part)
            progress.progress(min(1.0, (i+1) / num_chunks))
        
        progress.empty()
        status_text.empty()
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
            
            # Build variable selections - use first value for each variable
            variable_selections = {}
            if "variables" in metadata:
                for var in metadata["variables"]:
                    if var.get("values"):
                        variable_selections[var["code"]] = [var["values"][0]]
            
            # Request data using the correct PXWeb format
            return fetch_stat_fin_table_data_with_query(db_id, table_id, variable_selections)
                
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

def parse_pxweb_json_data(pxweb_response: Dict) -> pd.DataFrame:
    """Parse PXWeb API JSON format response into a pandas DataFrame"""
    try:
        data_values = pxweb_response.get("data", [])
        columns_info = pxweb_response.get("columns", [])
        
        if not columns_info or not data_values:
            return pd.DataFrame()
        
        # Get column codes (dimension names)
        column_codes = [col.get("code") for col in columns_info]
        column_texts = [col.get("text") for col in columns_info]
        
        rows = []
        for item in data_values:
            row = {}
            
            # item["key"] contains the dimension values as a list of strings
            key_parts = item.get("key", [])
            data_vals = item.get("values", [])
            
            # Map key values to column names
            for i, key_val in enumerate(key_parts):
                if i < len(column_codes):
                    col_name = column_texts[i] if i < len(column_texts) else f"Column_{i}"
                    row[col_name] = key_val
            
            # Add the data values
            # The last column in the response is typically the content/value column
            if data_vals:
                row["Value"] = data_vals[0]
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error parsing data: {e}")
        import traceback
        traceback.print_exc()
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

# Display API limits in sidebar
if data_source == "Statistics Finland":
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🇫🇮 Statistics Finland API Limits")
        st.info(f"""
- **Max cells per query:** {SF_API_LIMITS["maxCells"]:,}
- **Max values per query:** {SF_API_LIMITS["maxValues"]:,}
- **Max API calls:** {SF_API_LIMITS["maxCalls"]} per {SF_API_LIMITS["timeWindow"]}s
- **CORS enabled:** {'Yes' if SF_API_LIMITS["CORS"] else 'No'}
        """)

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
            
            if tables and isinstance(tables, list):
                # Create a mapping of table names to IDs
                table_dict = {}
                for item in tables:
                    if item.get("type") == "t":  # 't' = table, 'l' = folder/link
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
                        is_valid, size_msg = validate_query_size(variable_selections)
                        
                        # Display size validation
                        if not is_valid:
                            st.error(size_msg)
                        else:
                            st.info(size_msg)
                        
                        # Show rate limit status
                        remaining_calls = sf_rate_limiter.get_remaining_calls()
                        st.write(f"API calls remaining (60s window): **{remaining_calls}/{SF_API_LIMITS['maxCalls']}**")
                        
                        large_threshold = SF_API_LIMITS["maxCells"] // 4  # Warn at 1/4 of max
                        if estimated_cells > large_threshold:
                            st.warning("This query is large. Consider loading a preview or using chunked loading.")
                        # Chunked loading options
                        chunk_opts = st.expander("Chunked loading options (advanced)")
                        with chunk_opts:
                            max_chunk_cells = st.number_input("Max cells per request (chunk size):", min_value=500, max_value=20000, value=2000, step=500)
                            enable_chunked = st.checkbox("Enable chunked loading for large queries", value=True)

                        # Preview button: fetch a tiny preview using first value of each variable
                        if st.button("Load Preview (small)", key="load_preview"):
                            preview_selections = {code: [vals[0]] for code, vals in variable_selections.items() if vals}
                            preview_df = fetch_stat_fin_table_data_with_query(selected_db_id, selected_table_id, preview_selections)
                            if preview_df is not None:
                                st.dataframe(preview_df.head(100))

                        # Build and run the full query when requested
                        if st.button("Load Data", key="load_sf_data"):
                            if not is_valid:
                                st.error("Cannot load data: query exceeds API cell limit. Please reduce your selections.")
                            elif not variable_selections:
                                st.info("No variable selections available for this table. Attempting to load default data.")
                                df = fetch_stat_fin_table_data_with_query(selected_db_id, selected_table_id, {})
                            else:
                                # If estimated size is big and chunked loading enabled, use chunked_fetch
                                if estimated_cells > large_threshold and enable_chunked:
                                    st.info("Loading data in chunks...")
                                    df = chunked_fetch(selected_db_id, selected_table_id, variable_selections, max_chunk_cells=int(max_chunk_cells))
                                else:
                                    df = fetch_stat_fin_table_data_with_query(selected_db_id, selected_table_id, variable_selections)
                elif tables:
                    st.warning(f"Database has {len(tables)} items but no tables (type='t'). These may be folders or invalid entries.")
                    st.info(f"Response types found: {set(item.get('type') for item in tables)}")
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
    if show_chart:
        st.subheader("Visualization")

        chart_controls, chart_preview = st.columns([1, 2])

        # Detect column types
        numeric_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_columns = filtered_df.columns.tolist()

        with chart_controls:
            # Presets
            presets = ['None', 'Time Series', 'Comparison (grouped)', 'Distribution Dashboard']
            preset = st.selectbox("Preset:", options=presets, index=0)

            # Time-series detection
            detected_time_col = None
            if preset == 'Time Series':
                # Try to auto-detect a time column
                for c in all_columns:
                    if pd.api.types.is_datetime64_any_dtype(filtered_df[c]):
                        detected_time_col = c
                        break
                if detected_time_col is None:
                    # try parseable object columns
                    for c in all_columns:
                        if filtered_df[c].dtype == object:
                            try:
                                parsed = pd.to_datetime(filtered_df[c], errors='coerce')
                                if parsed.notna().sum() > 0:
                                    detected_time_col = c
                                    break
                            except Exception:
                                continue

                if detected_time_col:
                    st.markdown(f"Detected time column: **{detected_time_col}**")
                    use_time = st.checkbox("Use detected time column", value=True)
                    if use_time:
                        x_col = detected_time_col
                        # convert in-place for plotting
                        try:
                            filtered_df[x_col] = pd.to_datetime(filtered_df[x_col], errors='coerce')
                        except Exception:
                            pass

                    # Resampling options
                    resample_map = {'Daily':'D','Weekly':'W','Monthly':'M','Quarterly':'Q','Yearly':'Y'}
                    resample_label = st.selectbox('Resample frequency:', options=list(resample_map.keys()), index=2)
                    resample_freq = resample_map[resample_label]
                else:
                    st.info('No time-like column detected; pick X column manually.')

            # Dashboard templates (pre-made, one-click apply)
            dashboard_templates = {
                'None': [],
                'Tourism Overview': [
                    {'graph_type':'Line','x':'Year','y':'Value','title':'Visitors over time'},
                    {'graph_type':'Bar','x':'Region','y':'Value','agg':'sum','title':'Visitors by region'},
                    {'graph_type':'Pie','x':'Purpose','title':'Visit purpose share'}
                ],
                'Accommodation Summary': [
                    {'graph_type':'Line','x':'Year','y':'Nights','title':'Nights stayed (time series)'},
                    {'graph_type':'Bar','x':'AccommodationType','y':'Nights','agg':'sum','title':'Nights by accommodation type'},
                    {'graph_type':'Box','x':'Region','y':'Nights','title':'Nights distribution by region'}
                ],
                'Air Transport Snapshot': [
                    {'graph_type':'Line','x':'Year','y':'Passengers','title':'Air passengers over time'},
                    {'graph_type':'Bar','x':'Airport','y':'Passengers','agg':'sum','title':'Passengers by airport'},
                    {'graph_type':'Histogram','x':'DelayMinutes','title':'Delay distribution'}
                ],
                'Population Trends': [
                    {'graph_type':'Line','x':'Year','y':'Population','title':'Population over time'},
                    {'graph_type':'Bar','x':'AgeGroup','y':'Population','agg':'sum','title':'Population by age group'}
                ]
            }

            if preset != 'None':
                st.markdown(f"Preset layout: **{preset}**")

            def apply_dashboard_template(template_name, source_df):
                """Apply a preset template by building the listed charts from source_df
                and storing them in `st.session_state['dashboard_figs']` as serializable specs.
                """
                specs = dashboard_templates.get(template_name) or []
                if not specs:
                    return False, 'No specs for template'

                st.session_state['dashboard_figs'] = []
                for s in specs:
                    try:
                        gt = s.get('graph_type')
                        title = s.get('title') or f"{gt}"
                        x = s.get('x')
                        y = s.get('y')
                        agg = s.get('agg')
                        # Skip if required columns are missing
                        if x and x not in source_df.columns:
                            continue
                        if y and y not in source_df.columns:
                            continue

                        if gt == 'Line':
                            f = px.line(source_df, x=x, y=y, title=title)
                        elif gt == 'Bar':
                            if agg and y:
                                g = source_df.groupby(x)[y].agg(agg).reset_index()
                                f = px.bar(g, x=x, y=y, title=title)
                            else:
                                f = px.bar(source_df, x=x, y=y, title=title)
                        elif gt == 'Histogram':
                            f = px.histogram(source_df, x=x, title=title)
                        elif gt == 'Box':
                            f = px.box(source_df, x=x, y=y, title=title)
                        elif gt == 'Pie':
                            if y:
                                g = source_df.groupby(x)[y].sum().reset_index()
                                f = px.pie(g, names=x, values=y, title=title)
                            else:
                                g = source_df[x].value_counts().reset_index()
                                g.columns = [x, 'count']
                                f = px.pie(g, names=x, values='count', title=title)
                        else:
                            f = px.histogram(source_df, x=x, title=title)

                        st.session_state['dashboard_figs'].append({'title': title, 'fig': f, 'spec': s})
                    except Exception:
                        continue
                return True, f'Applied {len(st.session_state.get("dashboard_figs", []))} charts'

            # Template gallery UI: choose and apply a pre-made dashboard
            template_choice = st.selectbox('Dashboard template:', options=list(dashboard_templates.keys()), index=0)
            if template_choice and template_choice != 'None':
                st.markdown(f"**Selected template:** {template_choice}")
                if st.button('Apply template', key='apply_template'):
                    ok, msg = apply_dashboard_template(template_choice, filtered_df)
                    if ok:
                        st.success(msg)
                    else:
                        st.info(msg)

            # Aggregation heuristic for bars
            def suggest_agg(col_name, series):
                name = str(col_name).lower()
                if any(k in name for k in ['count', 'number', 'total', 'sum']):
                    return 'sum'
                if pd.api.types.is_integer_dtype(series):
                    return 'sum'
                if pd.api.types.is_float_dtype(series):
                    return 'mean'
                return 'count'
            graph_type = st.selectbox("Graph type:", options=[
                'Scatter', 'Line', 'Bar', 'Histogram', 'Box', 'Pie', 'Heatmap'
            ], index=0)

            x_col = st.selectbox("X column:", options=all_columns, index=0)
            # Y only meaningful for most charts
            y_col = None
            if graph_type in ['Scatter', 'Line', 'Bar', 'Box']:
                y_candidates = numeric_columns if numeric_columns else all_columns
                y_col = st.selectbox("Y column:", options=y_candidates, index=0, key='y_col')

            color_col = st.selectbox("Color (optional):", options=[None] + categorical_columns, index=0)
            agg_func = None
            if graph_type in ['Bar']:
                agg_func = st.selectbox("Aggregation:", options=['sum', 'mean', 'median', 'count'], index=0)

            log_scale = st.checkbox("Log scale (Y)", value=False)
            normalize = st.checkbox("Normalize (proportion) - Pie only", value=False)

            st.markdown("---")
            chart_title = st.text_input("Chart title:", value=f"{graph_type} of data")

        # Build and render the figure
        with chart_preview:
            try:
                fig = None
                if graph_type == 'Scatter':
                    fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col, title=chart_title)
                elif graph_type == 'Line':
                    fig = px.line(filtered_df, x=x_col, y=y_col, color=color_col, title=chart_title)
                elif graph_type == 'Bar':
                    if agg_func and y_col:
                        g = filtered_df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                        fig = px.bar(g, x=x_col, y=y_col, color=color_col, title=chart_title)
                    else:
                        fig = px.bar(filtered_df, x=x_col, y=y_col, color=color_col, title=chart_title)
                elif graph_type == 'Histogram':
                    # histogram only needs x
                    fig = px.histogram(filtered_df, x=x_col, color=color_col, title=chart_title)
                elif graph_type == 'Box':
                    fig = px.box(filtered_df, x=x_col, y=y_col, color=color_col, title=chart_title)
                elif graph_type == 'Pie':
                    # For pie, use counts of x_col or aggregation over y_col
                    if y_col and normalize:
                        g = filtered_df.groupby(x_col)[y_col].sum().reset_index()
                        fig = px.pie(g, names=x_col, values=y_col, title=chart_title)
                    else:
                        g = filtered_df[x_col].value_counts().reset_index()
                        g.columns = [x_col, 'count']
                        fig = px.pie(g, names=x_col, values='count', title=chart_title)
                elif graph_type == 'Heatmap':
                    # Try to create pivot table heatmap if possible
                    if y_col and y_col in numeric_columns:
                        pivot = filtered_df.pivot_table(index=y_col, columns=x_col, values=y_col, aggfunc='mean')
                        fig = px.imshow(pivot, title=chart_title)
                    else:
                        # fallback density heatmap
                        fig = px.density_heatmap(filtered_df, x=x_col, y=y_col if y_col else x_col, title=chart_title)

                if fig is not None:
                    if log_scale and hasattr(fig, 'update_yaxes'):
                        fig.update_yaxes(type='log')
                    st.plotly_chart(fig, use_container_width=True)

                    # Enhanced export & dashboard options
                    exp_col1, exp_col2 = st.columns([1,1])

                    # Initialize dashboard storage
                    if 'dashboard_figs' not in st.session_state:
                        st.session_state['dashboard_figs'] = []

                    # Add to dashboard
                    add_key = f"add_dashboard_{len(st.session_state['dashboard_figs'])}"
                    if exp_col1.button('Add chart to dashboard', key=add_key):
                        # store minimal serializable data: title + figure (Plotly fig is serializable)
                        st.session_state['dashboard_figs'].append({'title': chart_title, 'fig': fig})
                        st.success('Chart added to dashboard')

                    # Single-chart exports
                    with exp_col2:
                        try:
                            if st.button('Download PNG (chart)', key=f"png_chart_{len(st.session_state['dashboard_figs'])}"):
                                img_bytes = fig.to_image(format='png')
                                st.download_button('Save PNG', data=img_bytes, file_name='chart.png', mime='image/png')
                        except Exception:
                            st.warning('PNG export for single chart requires the `kaleido` package.')

                        html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                        if st.button('Download HTML Embed', key=f"html_chart_{len(st.session_state['dashboard_figs'])}"):
                            st.download_button('Save HTML', data=html, file_name='chart_embed.html', mime='text/html')
                        st.markdown('**Embed snippet:**')
                        st.code(html, language='html')

                    # Dashboard manager
                    if st.session_state.get('dashboard_figs'):
                        with st.expander('Dashboard (' + str(len(st.session_state['dashboard_figs'])) + ' charts)'):
                            for i, item in enumerate(st.session_state['dashboard_figs']):
                                st.write(f"{i+1}. {item['title']}")

                            if st.button('Clear dashboard', key='clear_dashboard'):
                                st.session_state['dashboard_figs'] = []
                                st.experimental_rerun()

                            # Export dashboard as HTML or PNG; also save/load templates
                            parts = [d['fig'].to_html(full_html=False, include_plotlyjs='cdn') for d in st.session_state['dashboard_figs']]
                            full_html = "<html><head><meta charset='utf-8'></head><body>" + "<hr/>".join(parts) + "</body></html>"
                            if st.button('Download dashboard HTML', key='export_dashboard_html'):
                                st.download_button('Download dashboard HTML', data=full_html, file_name='dashboard.html', mime='text/html')

                            # Template load/save
                            uploaded = st.file_uploader('Load dashboard template (JSON)', type=['json'])
                            if uploaded is not None:
                                try:
                                    j = json.load(uploaded)
                                    st.session_state['dashboard_figs'] = []
                                    for s in j:
                                        gt = s.get('graph_type')
                                        try:
                                            if gt == 'Line':
                                                f = px.line(filtered_df, x=s.get('x'), y=s.get('y'), title=s.get('title'))
                                            elif gt == 'Histogram':
                                                f = px.histogram(filtered_df, x=s.get('x'), title=s.get('title'))
                                            elif gt == 'Bar':
                                                if s.get('agg') and s.get('y'):
                                                    g = filtered_df.groupby(s.get('x'))[s.get('y')].agg(s.get('agg')).reset_index()
                                                    f = px.bar(g, x=s.get('x'), y=s.get('y'), title=s.get('title'))
                                                else:
                                                    f = px.bar(filtered_df, x=s.get('x'), y=s.get('y'), title=s.get('title'))
                                            elif gt == 'Box':
                                                f = px.box(filtered_df, x=s.get('x'), y=s.get('y'), title=s.get('title'))
                                            else:
                                                f = px.histogram(filtered_df, x=s.get('x'), title=s.get('title'))
                                        except Exception:
                                            continue
                                        st.session_state['dashboard_figs'].append({'title': s.get('title'), 'fig': f, 'spec': s})
                                    st.success('Template loaded')
                                except Exception as e:
                                    st.error(f'Failed to load template: {e}')

                            if st.button('Save dashboard as template (JSON)', key='save_template'):
                                specs = [d.get('spec') for d in st.session_state['dashboard_figs'] if d.get('spec')]
                                if specs:
                                    st.download_button('Download template', data=json.dumps(specs, indent=2), file_name='dashboard_template.json', mime='application/json')
                                else:
                                    st.info('No specs available to save')

                            # Grid layout options for stitched PNG
                            cols = st.number_input('Grid columns for PNG export', min_value=1, max_value=6, value=1)
                            if st.button('Export dashboard as PNG', key='export_dashboard_png'):
                                pngs = []
                                failed = False
                                for d in st.session_state['dashboard_figs']:
                                    try:
                                        b = d['fig'].to_image(format='png')
                                        pngs.append(Image.open(io.BytesIO(b)))
                                    except Exception:
                                        failed = True
                                        break

                                if failed or not pngs:
                                    st.warning('PNG export of full dashboard requires `kaleido` and `Pillow`.')
                                else:
                                    n = len(pngs)
                                    grid_cols = int(cols)
                                    grid_rows = math.ceil(n / grid_cols)
                                    maxw = max(im.width for im in pngs)
                                    maxh = max(im.height for im in pngs)
                                    dst = Image.new('RGBA', (grid_cols * maxw, grid_rows * maxh), (255,255,255,255))
                                    for idx, im in enumerate(pngs):
                                        row = idx // grid_cols
                                        col = idx % grid_cols
                                        x = col * maxw + (maxw - im.width) // 2
                                        y = row * maxh + (maxh - im.height) // 2
                                        dst.paste(im, (x, y))
                                    buf = io.BytesIO()
                                    dst.convert('RGB').save(buf, format='PNG')
                                    buf.seek(0)
                                    st.download_button('Download dashboard PNG', data=buf, file_name='dashboard.png', mime='image/png')
                else:
                    st.info("Unable to build chart with the selected options.")
            except Exception as e:
                st.error(f"Error building chart: {e}")

else:
    if data_source != "Sample Data":
        st.warning("No data available. Please upload a file or select Sample Data.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/)")
