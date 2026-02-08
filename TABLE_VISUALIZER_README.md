# Table Data Visualizer

A Python web application built with Streamlit that visualizes table-like data with interactive filtering, statistics, and charts.

## Features

✨ **Interactive Features:**
- Load sample data, upload CSV files, or manually input data
- Real-time filtering by column values
- Interactive data tables with sorting and searching
- Statistics display (mean, median, std dev, etc.)
- Dynamic chart generation (scatter plots, histograms)
- Responsive design that works on desktop and mobile

## Installation

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Locally

Run the Streamlit app:
```bash
streamlit run table_visualizer.py
```

The app will open in your browser at `http://localhost:8501`

### Embedding in Your Website

#### Option 1: Embed via iframe (Recommended)

1. Deploy the app to a service like:
   - [Streamlit Cloud](https://streamlit.io/cloud) (free tier available)
   - [Heroku](https://www.heroku.com/)
   - Your own server

2. Add an iframe to your HTML:
   ```html
   <iframe 
     src="https://your-deployed-app-url.streamlit.app" 
     frameborder="0" 
     height="800" 
     width="100%">
   </iframe>
   ```

#### Option 2: Self-hosted with custom domain

You can run it on your own server and access it via:
```
https://yourdomain.com/visualizer
```

## Usage

1. **Select Data Source** (in sidebar):
   - **Sample Data**: Use pre-loaded example dataset
   - **Upload CSV**: Load your own CSV file
   - **Manual Input**: Create data directly in the app

2. **Enable/Disable Features**:
   - Toggle Statistics display
   - Toggle Chart visualization
   - Toggle Filters

3. **Filter Data**:
   - Select columns to filter by
   - Choose specific values or ranges
   - Results update in real-time

4. **Analyze Data**:
   - View summary statistics
   - Create custom scatter plots and histograms
   - Export data as CSV

## Customization

### Change Sample Data

Edit the sample data in `table_visualizer.py` (lines ~30-40):
```python
df = pd.DataFrame({
    'Your Column': [values...],
    # Add more columns as needed
})
```

### Change Title and Styling

Modify the Streamlit configuration:
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="📊",
    layout="wide",
)
```

### Add More Visualizations

Add custom charts using Plotly:
```python
import plotly.express as px
fig = px.box(df, y='column_name', title='Box Plot')
st.plotly_chart(fig, use_container_width=True)
```

## Deployment Guide

### Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Select the Python file to run
5. Your app is live!

### Deploy to Your Own Server

```bash
# Install Streamlit on server
pip install streamlit pandas plotly

# Run with production settings
streamlit run table_visualizer.py \
  --server.port=80 \
  --server.address=0.0.0.0 \
  --logger.level=error
```

## File Structure

```
JoseLahtinen.github.io/
├── table_visualizer.py      # Main Streamlit application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Browser Compatibility

Works with:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Troubleshooting

**App won't start:**
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Try updating Streamlit
pip install --upgrade streamlit
```

**CSV not loading:**
- Ensure CSV is UTF-8 encoded
- Check column names don't have special characters
- Verify data format is correct

**Slow performance with large datasets:**
- Filter data in your CSV before uploading
- Consider using Streamlit's caching:
```python
@st.cache_data
def load_data(file):
    return pd.read_csv(file)
```

## License

This project is open source and available for personal and commercial use.

## Support

For issues with:
- **Streamlit**: [streamlit.io/docs](https://docs.streamlit.io/)
- **Pandas**: [pandas.pydata.org](https://pandas.pydata.org/)
- **Plotly**: [plotly.com](https://plotly.com/python/)
