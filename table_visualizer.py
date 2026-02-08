import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from typing import Dict, List, Optional

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
            response = requests.get(SF_API_BASE, timeout=10)
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
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []

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
                    category_idx = int(key_parts[i])
                    categories = dim_info.get("category", {})
                    if "index" in categories:
                        idx_list = categories["index"].get(str(category_idx), [])
                        if isinstance(idx_list, list) and idx_list:
                            row[dim_name] = idx_list[0]
                        elif isinstance(idx_list, (int, str)):
                            row[dim_name] = idx_list
                    elif "label" in categories:
                        row[dim_name] = categories["label"].get(str(category_idx), f"Category {category_idx}")
            
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
        'In Stock': [Yes := True, Yes, No := False, Yes, Yes, No]
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
                        
                        # Load the data
                        if st.button("Load Data", key="load_sf_data"):
                            df = fetch_stat_fin_table_data(selected_db_id, selected_table_id)
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
